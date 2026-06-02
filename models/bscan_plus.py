import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import Classifier


class ResidualLocalBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                d_model,
                d_model,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=d_model,
            ),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.Dropout(dropout),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class BSCANPlus(nn.Module):
    """
    BSCAN+: token CNN encoder with explicit interpretable RC stem features.

    Inputs:
      upper_seq [B, 2L] = upper intron + upper exon token ids
      lower_seq [B, 2L] = lower exon + lower intron token ids

    The stem branch exposes:
      - WC stem map after reverse-complementing lower_seq
      - GU wobble-compatible map
      - per-position pairing profile
      - short continuous-stem scores from diagonal filters
    """

    def __init__(
        self,
        junction_bps: int = 100,
        length_seq: int | None = None,
        d_model: int = 128,
        conv_channels: int = 256,
        dropout_prob: float = 0.25,
        local_kernel_size: int = 9,
        local_dilations: tuple[int, ...] = (1, 2, 4),
        stem_windows: tuple[int, ...] = (4, 6, 8),
        wobble_weight: float = 0.5,
        use_cross_gate: bool = True,
    ):
        super().__init__()
        self.junction_bps = junction_bps
        self.length_seq = length_seq or (2 * junction_bps)
        self.stem_windows = stem_windows
        self.wobble_weight = wobble_weight
        self.use_cross_gate = use_cross_gate
        self.use_pretrained = False

        self.embeddings = nn.Embedding(26, d_model)
        nn.init.normal_(self.embeddings.weight, std=0.02)

        self.local_encoder = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=local_kernel_size, padding=(local_kernel_size - 1) // 2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            *[
                ResidualLocalBlock(d_model, local_kernel_size, dilation, dropout_prob)
                for dilation in local_dilations
            ],
        )
        self.proj = nn.Sequential(
            nn.Conv1d(d_model, conv_channels, kernel_size=1),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveMaxPool1d(self.length_seq // 16)
        self.drop = nn.Dropout(dropout_prob)

        if use_cross_gate:
            self.cross_gate = nn.Sequential(
                nn.Linear(8 * conv_channels, conv_channels),
                nn.GELU(),
                nn.Dropout(dropout_prob),
            )
            cross_dim = conv_channels
        else:
            self.cross_gate = None
            cross_dim = 0

        stem_profile_dim = 2 * junction_bps
        stem_stats_dim = 8 + (2 * len(stem_windows))
        seq_stats_dim = 20
        pooled_dim = 2 * conv_channels * (self.length_seq // 16)
        self.fc_input_dim = pooled_dim + cross_dim + stem_profile_dim + stem_stats_dim + seq_stats_dim
        self.classifier = Classifier(d_in=self.fc_input_dim, dropout=dropout_prob)

        self.base_token_ids = {6: 0, 7: 1, 8: 2, 9: 3}
        rc_lookup = torch.arange(26, dtype=torch.long)
        rc_lookup[6] = 9
        rc_lookup[9] = 6
        rc_lookup[7] = 8
        rc_lookup[8] = 7
        self.register_buffer("rc_lookup", rc_lookup, persistent=False)

    def _reverse_complement_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.rc_lookup[tokens.flip(dims=[1])]

    def _token_to_one_hot(self, tokens: torch.Tensor) -> torch.Tensor:
        x = torch.zeros(tokens.size(0), 4, tokens.size(1), device=tokens.device)
        for token_id, channel in self.base_token_ids.items():
            x[:, channel, :] = (tokens == token_id).to(x.dtype)
        return x

    def _encode_features(self, tokens: torch.Tensor):
        emb = self.embeddings(tokens).transpose(1, 2)
        local = self.local_encoder(emb)
        feat_map = self.proj(local)
        pooled = self.pool(feat_map).flatten(start_dim=1)
        summary = torch.cat(
            [feat_map.mean(dim=2), feat_map.amax(dim=2)],
            dim=1,
        )
        return self.drop(pooled), summary

    def _wobble_map(self, upper_intron: torch.Tensor, lower_intron_rc: torch.Tensor) -> torch.Tensor:
        # In RC coordinates, original G-U/U-G wobble appears as G-A or U-C.
        g_a = (upper_intron == 8).unsqueeze(2) & (lower_intron_rc == 6).unsqueeze(1)
        u_c = (upper_intron == 9).unsqueeze(2) & (lower_intron_rc == 7).unsqueeze(1)
        return (g_a | u_c).to(torch.float32)

    def _continuity_stats(self, stem_map: torch.Tensor) -> torch.Tensor:
        x = stem_map.unsqueeze(1)
        stats = []
        for window in self.stem_windows:
            kernel = torch.eye(window, device=stem_map.device, dtype=stem_map.dtype).view(1, 1, window, window)
            diag = F.conv2d(x, kernel) / float(window)
            stats.extend([diag.amax(dim=(1, 2, 3)), diag.mean(dim=(1, 2, 3))])
        return torch.stack(stats, dim=1)

    def _stem_features(self, upper_seq: torch.Tensor, lower_rc_seq: torch.Tensor):
        L = self.junction_bps
        upper_intron = upper_seq[:, :L]
        lower_intron_rc = lower_rc_seq[:, :L]

        u_oh = self._token_to_one_hot(upper_intron)
        l_rc_oh = self._token_to_one_hot(lower_intron_rc)
        wc_map = torch.bmm(u_oh.transpose(1, 2), l_rc_oh)
        wobble_map = self._wobble_map(upper_intron, lower_intron_rc)
        stem_map = torch.clamp(wc_map + self.wobble_weight * wobble_map, max=1.0)

        row_max = stem_map.max(dim=-1).values
        col_max = stem_map.max(dim=-2).values
        wc_row_max = wc_map.max(dim=-1).values
        wobble_row_max = wobble_map.max(dim=-1).values
        continuity = self._continuity_stats(stem_map)

        stats = torch.cat(
            [
                wc_map.mean(dim=(1, 2), keepdim=False).unsqueeze(1),
                stem_map.mean(dim=(1, 2), keepdim=False).unsqueeze(1),
                wc_map.amax(dim=(1, 2), keepdim=False).unsqueeze(1),
                stem_map.amax(dim=(1, 2), keepdim=False).unsqueeze(1),
                row_max.mean(dim=1, keepdim=True),
                col_max.mean(dim=1, keepdim=True),
                wc_row_max.mean(dim=1, keepdim=True),
                wobble_row_max.mean(dim=1, keepdim=True),
                continuity,
            ],
            dim=1,
        )
        stem_profile = torch.cat([row_max, col_max], dim=1)
        return self.drop(torch.cat([stem_profile, stats], dim=1)), wc_map, wobble_map, stem_map, stats

    def _sequence_stats(self, upper_seq: torch.Tensor, lower_seq: torch.Tensor) -> torch.Tensor:
        L = self.junction_bps
        regions = [
            upper_seq[:, :L],
            upper_seq[:, L:],
            lower_seq[:, :L],
            lower_seq[:, L:],
        ]
        parts = []
        for region in regions:
            base_freq = self._token_to_one_hot(region).mean(dim=2)
            gc_frac = base_freq[:, 1:3].sum(dim=1, keepdim=True)
            parts.extend([base_freq, gc_frac])
        return torch.cat(parts, dim=1)

    def extract_features(self, upper_seq: torch.Tensor, lower_seq: torch.Tensor):
        lower_rc_seq = self._reverse_complement_tokens(lower_seq)
        upper_pooled, upper_summary = self._encode_features(upper_seq)
        lower_pooled, lower_summary = self._encode_features(lower_seq)
        stem_feat, wc_map, wobble_map, stem_map, stem_stats = self._stem_features(upper_seq, lower_rc_seq)
        seq_stats = self._sequence_stats(upper_seq, lower_seq)

        parts = [upper_pooled, lower_pooled]
        cross_feat = None
        if self.use_cross_gate:
            cross_feat = self.cross_gate(
                torch.cat(
                    [
                        upper_summary,
                        lower_summary,
                        torch.abs(upper_summary - lower_summary),
                        upper_summary * lower_summary,
                    ],
                    dim=1,
                )
            )
            parts.append(cross_feat)
        parts.extend([stem_feat, seq_stats])
        features = torch.cat(parts, dim=1)

        aux = {
            "bp_map": wc_map.detach(),
            "wobble_map": wobble_map.detach(),
            "stem_map": stem_map.detach(),
            "stem_stats": stem_stats.detach(),
            "lower_rc_seq": lower_rc_seq.detach(),
        }
        if cross_feat is not None:
            aux["cross_gate"] = cross_feat.detach()
        return features, aux

    def forward(self, upper_seq: torch.Tensor, lower_seq: torch.Tensor, return_aux: bool = False):
        features, aux = self.extract_features(upper_seq, lower_seq)
        logits = self.classifier(features)
        if not return_aux:
            return logits
        return logits, aux

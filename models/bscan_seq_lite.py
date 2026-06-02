import torch
import torch.nn as nn
from multimolecule import RnaBertModel, RnaErnieModel

from .classifier import Classifier


class BSCANSeqLite(nn.Module):
    """
    BSCAN-Seq-Lite: CNN backbone plus an explicit intronic RC stem profile.

    Inputs:
      upper_seq [B, 2L] = upper intron + upper exon token ids
      lower_seq [B, 2L] = lower exon + lower intron token ids

    The model computes RC(lower_seq) internally and adds a simple
    Watson-Crick intron pairing profile to a shared upper/lower CNN backbone.
    It intentionally removes attention and 2D stem CNN blocks.
    """

    def __init__(
        self,
        junction_bps: int = 100,
        d_model: int = 128,
        conv_out_channels1: int = 256,
        conv_kernel_size1: int = 11,
        conv_out_channels2: int = 256,
        conv_kernel_size2: int = 11,
        length_seq: int | None = None,
        dropout_prob: float = 0.2,
        pretrained_model: str = "rnaernie",
        use_pretrained: bool = False,
        use_stem_profile: bool = True,
        use_stem_stats: bool = True,
        use_seq_stats: bool = False,
        use_cross_attention: bool = False,
        cross_attention_heads: int = 4,
    ):
        super().__init__()
        self.junction_bps = junction_bps
        self.length_seq = length_seq or (2 * junction_bps)
        self.use_pretrained = use_pretrained
        self.use_stem_profile = use_stem_profile
        self.use_stem_stats = use_stem_stats
        self.use_seq_stats = use_seq_stats
        self.use_cross_attention = use_cross_attention

        if use_pretrained:
            if pretrained_model.lower() == "rnaernie":
                pretrained = RnaErnieModel.from_pretrained("multimolecule/rnaernie")
            elif pretrained_model.lower() == "rnabert":
                pretrained = RnaBertModel.from_pretrained("multimolecule/rnabert")
            else:
                raise ValueError(f"Invalid pretrained model: {pretrained_model}")
            self.embeddings = pretrained.embeddings
            self.encoder = None
            for param in self.embeddings.parameters():
                param.requires_grad = False
            self.to_proj = nn.Sequential(nn.Linear(768, d_model), nn.GELU())
        else:
            self.embeddings = nn.Embedding(26, d_model)
            nn.init.normal_(self.embeddings.weight, std=0.02)
            self.to_proj = nn.Identity()

        if self.use_cross_attention:
            if d_model % cross_attention_heads != 0:
                raise ValueError(
                    f"d_model ({d_model}) must be divisible by cross_attention_heads ({cross_attention_heads})."
                )
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=cross_attention_heads,
                batch_first=True,
            )
            self.cross_attention_norm = nn.LayerNorm(2 * d_model)
        else:
            self.cross_attention = None
            self.cross_attention_norm = None

        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(
                d_model,
                conv_out_channels1,
                conv_kernel_size1,
                padding=(conv_kernel_size1 - 1) // 2,
            ),
            nn.BatchNorm1d(conv_out_channels1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(
                conv_out_channels1,
                conv_out_channels2,
                conv_kernel_size2,
                padding=(conv_kernel_size2 - 1) // 2,
            ),
            nn.BatchNorm1d(conv_out_channels2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )
        self.drop = nn.Dropout(dropout_prob)
        self.out_dim = self.length_seq // (4 * 4)

        profile_dim = 2 * junction_bps if use_stem_profile else 0
        stats_dim = 4 if use_stem_stats else 0
        seq_stats_dim = 20 if use_seq_stats else 0
        cross_attn_dim = 2 * d_model if use_cross_attention else 0
        self.fc_input_dim = (
            conv_out_channels2 * self.out_dim * 2
        ) + profile_dim + stats_dim + seq_stats_dim
        self.fc_input_dim += cross_attn_dim
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

    def _encode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.to_proj(self.embeddings(tokens))

    def _cnn_features(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self._encode(tokens)
        x = self.conv_layer1(embedded.transpose(1, 2))
        x = self.conv_layer2(x)
        return self.drop(x.flatten(start_dim=1))

    def _token_to_one_hot(self, tokens: torch.Tensor) -> torch.Tensor:
        x = torch.zeros(tokens.size(0), 4, tokens.size(1), device=tokens.device)
        for token_id, channel in self.base_token_ids.items():
            x[:, channel, :] = (tokens == token_id).to(x.dtype)
        return x

    def _stem_profile(self, upper_seq: torch.Tensor, lower_rc_seq: torch.Tensor):
        L = self.junction_bps
        upper_intron = upper_seq[:, :L]
        lower_intron_rc = lower_rc_seq[:, :L]

        u_oh = self._token_to_one_hot(upper_intron)
        l_rc_oh = self._token_to_one_hot(lower_intron_rc)
        bp_map = torch.bmm(u_oh.transpose(1, 2), l_rc_oh)

        row_max = bp_map.max(dim=-1).values
        col_max = bp_map.max(dim=-2).values
        stats = torch.stack(
            [
                bp_map.mean(dim=(1, 2)),
                bp_map.amax(dim=(1, 2)),
                row_max.mean(dim=1),
                col_max.mean(dim=1),
            ],
            dim=1,
        )

        parts = []
        if self.use_stem_profile:
            parts.extend([row_max, col_max])
        if self.use_stem_stats:
            parts.append(stats)
        stem_feat = torch.cat(parts, dim=1) if parts else bp_map.new_zeros(bp_map.size(0), 0)
        return self.drop(stem_feat), bp_map, row_max, col_max, stats

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

        upper_conv = self._cnn_features(upper_seq)
        lower_conv = self._cnn_features(lower_seq)
        stem_feat, bp_map, row_max, col_max, stats = self._stem_profile(upper_seq, lower_rc_seq)

        parts = [upper_conv, lower_conv]
        cross_attn_feat = None
        if self.use_cross_attention:
            upper_emb = self._encode(upper_seq)
            lower_rc_emb = self._encode(lower_rc_seq)
            upper_ctx, _ = self.cross_attention(
                upper_emb,
                lower_rc_emb,
                lower_rc_emb,
                need_weights=False,
            )
            lower_ctx, _ = self.cross_attention(
                lower_rc_emb,
                upper_emb,
                upper_emb,
                need_weights=False,
            )
            cross_attn_feat = self.cross_attention_norm(
                torch.cat([upper_ctx.mean(dim=1), lower_ctx.mean(dim=1)], dim=1)
            )
            parts.append(cross_attn_feat)

        parts.append(stem_feat)
        seq_stats = None
        if self.use_seq_stats:
            seq_stats = self._sequence_stats(upper_seq, lower_seq)
            parts.append(seq_stats)

        features = torch.cat(parts, dim=1)
        aux = {
            "bp_map": bp_map.detach(),
            "row_max": row_max.detach(),
            "col_max": col_max.detach(),
            "stem_stats": stats.detach(),
            "lower_rc_seq": lower_rc_seq.detach(),
        }
        if cross_attn_feat is not None:
            aux["cross_attention"] = cross_attn_feat.detach()
        if seq_stats is not None:
            aux["sequence_stats"] = seq_stats.detach()
        return features, aux

    def forward(self, upper_seq, lower_seq, return_aux: bool = False):
        features, aux = self.extract_features(upper_seq, lower_seq)
        logits = self.classifier(features)

        if not return_aux:
            return logits
        return logits, aux

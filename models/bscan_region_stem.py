import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import Classifier


_RC_PERM = [3, 2, 1, 0]  # repo one-hot order: A, G, C, T -> T, C, G, A


def _rc_onehot(x: torch.Tensor) -> torch.Tensor:
    return x[:, _RC_PERM, :].flip(dims=[2])


class _RegionEncoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 160,
        kernel_size: int = 9,
        pooled_len: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(pooled_len),
        )
        self.summary = nn.Sequential(
            nn.Linear(out_channels * pooled_len, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x).flatten(start_dim=1)
        return self.summary(x)


class BSCANRegionStem(nn.Module):
    """
    One-hot BSCAN variant with explicit region separation and stem features.

    Inputs:
      upper_seq [B, 4, 2L] = upper intron + upper exon
      lower_seq [B, 4, 2L] = lower exon + lower intron

    The model keeps the input representation strictly one-hot while adding:
      - separate encoders for four biological regions
      - exon-pair and intron-pair interaction summaries
      - WC/GU stem profiles and diagonal continuity scores
    """

    def __init__(
        self,
        junction_bps: int = 100,
        region_channels: int = 160,
        region_kernel: int = 9,
        region_pooled_len: int = 8,
        stem_channels: tuple[int, int, int] = (16, 32, 48),
        stem_adaptive: int = 4,
        stem_windows: tuple[int, ...] = (4, 6, 8),
        wobble_weight: float = 0.5,
        dropout: float = 0.25,
        d_hidden: int = 256,
    ):
        super().__init__()
        self.junction_bps = junction_bps
        self.stem_windows = stem_windows
        self.wobble_weight = wobble_weight

        self.region_encoder = _RegionEncoder(
            out_channels=region_channels,
            kernel_size=region_kernel,
            pooled_len=region_pooled_len,
            dropout=dropout,
        )
        d_region = self.region_encoder.out_dim

        c1, c2, c3 = stem_channels
        self.stem_cnn = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(stem_adaptive),
        )
        d_stem_map = c3 * stem_adaptive * stem_adaptive

        region_dim = 4 * d_region
        interaction_dim = 4 * d_region
        stem_profile_dim = 2 * junction_bps
        stem_stat_dim = 8 + 2 * len(stem_windows)
        d_in = region_dim + interaction_dim + stem_profile_dim + stem_stat_dim + d_stem_map
        self.drop = nn.Dropout(dropout)
        self.classifier = Classifier(d_in=d_in, d_hiddens=[d_hidden], dropout=dropout, use_layernorm=True)

    def _wobble_map(self, upper_intron: torch.Tensor, lower_intron_rc: torch.Tensor) -> torch.Tensor:
        # After reverse-complementing the lower intron, original GU/UG wobble
        # is represented as G-A or T-C. Repo one-hot order is A,G,C,T.
        u_g = upper_intron[:, 1, :].unsqueeze(2)
        u_t = upper_intron[:, 3, :].unsqueeze(2)
        l_a = lower_intron_rc[:, 0, :].unsqueeze(1)
        l_c = lower_intron_rc[:, 2, :].unsqueeze(1)
        return ((u_g * l_a) + (u_t * l_c)).clamp(max=1.0)

    def _continuity_stats(self, stem_map: torch.Tensor) -> torch.Tensor:
        x = stem_map.unsqueeze(1)
        stats = []
        for window in self.stem_windows:
            kernel = torch.eye(window, device=stem_map.device, dtype=stem_map.dtype).view(1, 1, window, window)
            diag = F.conv2d(x, kernel) / float(window)
            stats.extend([diag.amax(dim=(1, 2, 3)), diag.mean(dim=(1, 2, 3))])
        return torch.stack(stats, dim=1)

    def _stem_features(self, upper_intron: torch.Tensor, lower_intron: torch.Tensor):
        lower_intron_rc = _rc_onehot(lower_intron)
        wc_map = torch.bmm(upper_intron.transpose(1, 2), lower_intron_rc)
        wobble_map = self._wobble_map(upper_intron, lower_intron_rc)
        stem_map = torch.clamp(wc_map + self.wobble_weight * wobble_map, max=1.0)

        row_max = stem_map.max(dim=-1).values
        col_max = stem_map.max(dim=-2).values
        wc_row_max = wc_map.max(dim=-1).values
        wobble_row_max = wobble_map.max(dim=-1).values
        continuity = self._continuity_stats(stem_map)
        stem_stats = torch.cat(
            [
                wc_map.mean(dim=(1, 2), keepdim=True).flatten(1),
                stem_map.mean(dim=(1, 2), keepdim=True).flatten(1),
                wc_map.amax(dim=(1, 2), keepdim=True).flatten(1),
                stem_map.amax(dim=(1, 2), keepdim=True).flatten(1),
                row_max.mean(dim=1, keepdim=True),
                col_max.mean(dim=1, keepdim=True),
                wc_row_max.mean(dim=1, keepdim=True),
                wobble_row_max.mean(dim=1, keepdim=True),
                continuity,
            ],
            dim=1,
        )
        stem_map_feat = self.stem_cnn(stem_map.unsqueeze(1)).flatten(start_dim=1)
        return row_max, col_max, stem_stats, stem_map_feat, wc_map, wobble_map, stem_map

    @staticmethod
    def _pair_interaction(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.abs(a - b), a * b], dim=1)

    def forward(self, upper_seq: torch.Tensor, lower_seq: torch.Tensor, return_aux: bool = False):
        L = self.junction_bps
        upper_intron = upper_seq[:, :, :L]
        upper_exon = upper_seq[:, :, L:]
        lower_exon = lower_seq[:, :, :L]
        lower_intron = lower_seq[:, :, L:]

        ui = self.region_encoder(upper_intron)
        ue = self.region_encoder(upper_exon)
        le = self.region_encoder(lower_exon)
        li = self.region_encoder(lower_intron)

        exon_inter = self._pair_interaction(ue, le)
        intron_inter = self._pair_interaction(ui, li)
        row_max, col_max, stem_stats, stem_map_feat, wc_map, wobble_map, stem_map = self._stem_features(
            upper_intron, lower_intron
        )

        features = torch.cat(
            [
                ui,
                ue,
                le,
                li,
                exon_inter,
                intron_inter,
                row_max,
                col_max,
                stem_stats,
                stem_map_feat,
            ],
            dim=1,
        )
        logits = self.classifier(self.drop(features))
        if not return_aux:
            return logits
        return logits, {
            "region_features": {
                "upper_intron": ui.detach(),
                "upper_exon": ue.detach(),
                "lower_exon": le.detach(),
                "lower_intron": li.detach(),
            },
            "wc_map": wc_map.detach(),
            "wobble_map": wobble_map.detach(),
            "stem_map": stem_map.detach(),
            "row_max": row_max.detach(),
            "col_max": col_max.detach(),
            "stem_stats": stem_stats.detach(),
        }

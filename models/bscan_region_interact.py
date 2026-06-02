import torch
import torch.nn as nn

from .classifier import Classifier
from .bscan_seq_lite import BSCANSeqLite


class _PairEncoder(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.GELU(),
        )

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        x = torch.cat([u, v, u * v, torch.abs(u - v)], dim=1)
        return self.net(x)


class BSCANRegionInteract(BSCANSeqLite):
    """
    BSCAN-Region:
    - keeps the strong BSCAN CNN + stem backbone
    - compresses each of the four junction regions into latent tokens
    - computes region-level pair interactions instead of token-level attention
    """

    def __init__(
        self,
        *args,
        region_hidden_dim: int = 128,
        pair_hidden_dim: int = 256,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        base_dim = self.conv_layer1[0].in_channels
        self.region_proj = nn.Sequential(
            nn.LayerNorm(2 * base_dim),
            nn.Linear(2 * base_dim, region_hidden_dim),
            nn.GELU(),
        )
        self.pair_encoder = _PairEncoder(base_dim, pair_hidden_dim, dropout=self.drop.p)

        self.region_dim = 4 * region_hidden_dim
        self.pair_dim = 4 * base_dim
        self.fc_input_dim = self.fc_input_dim + self.region_dim + self.pair_dim
        self.classifier = Classifier(d_in=self.fc_input_dim, dropout=self.drop.p)

    def _region_token(self, encoded: torch.Tensor, start: int, end: int) -> torch.Tensor:
        region = encoded[:, start:end, :]
        mean = region.mean(dim=1)
        mx = region.amax(dim=1)
        return self.region_proj(torch.cat([mean, mx], dim=1))

    def _region_pair_features(self, upper_seq: torch.Tensor, lower_rc_seq: torch.Tensor):
        upper_enc = self._encode(upper_seq)
        lower_rc_enc = self._encode(lower_rc_seq)
        L = self.junction_bps

        u_int = self._region_token(upper_enc, 0, L)
        u_ext = self._region_token(upper_enc, L, 2 * L)
        l_int = self._region_token(lower_rc_enc, 0, L)
        l_ext = self._region_token(lower_rc_enc, L, 2 * L)

        pair_feats = [
            self.pair_encoder(u_int, l_int),
            self.pair_encoder(u_ext, l_ext),
            self.pair_encoder(u_int, l_ext),
            self.pair_encoder(u_ext, l_int),
        ]
        region_feats = [u_int, u_ext, l_int, l_ext]
        return torch.cat(region_feats + pair_feats, dim=1), {
            "region_tokens": torch.cat(region_feats, dim=1).detach(),
            "pair_tokens": torch.cat(pair_feats, dim=1).detach(),
        }

    def extract_features(self, upper_seq: torch.Tensor, lower_seq: torch.Tensor):
        lower_rc_seq = self._reverse_complement_tokens(lower_seq)

        upper_conv = self._cnn_features(upper_seq)
        lower_conv = self._cnn_features(lower_seq)
        stem_feat, bp_map, row_max, col_max, stats = self._stem_profile(upper_seq, lower_rc_seq)
        region_pair_feat, region_aux = self._region_pair_features(upper_seq, lower_rc_seq)

        parts = [upper_conv, lower_conv, stem_feat, region_pair_feat]
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
            **region_aux,
        }
        if seq_stats is not None:
            aux["sequence_stats"] = seq_stats.detach()
        return features, aux

"""
BscanRegression: A version of the CircCombine model adapted for regression tasks,
such as predicting expression levels (e.g., read counts).

The only change from the original CircCombine (bscan) model is the final
classifier head, which is modified to output a single continuous value instead of
two class logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the same classifier but will initialize it with n_classes=1
from .classifier import Classifier

_RC_PERM = [3, 2, 1, 0]   # A=0,C=1,G=2,T=3  ->  T,G,C,A


def _rc(x: torch.Tensor) -> torch.Tensor:
    return x[:, _RC_PERM, :].flip(dims=[2])


def _bp_map(u_oh: torch.Tensor, l_rc_oh: torch.Tensor) -> torch.Tensor:
    """Hard WC base-pairing: [B, L, 4] @ [B, 4, L] -> [B, L, L]"""
    return torch.bmm(u_oh.transpose(1, 2), l_rc_oh)


# -- Sub-modules ---------------------------------------------------------------

class _CrossAttnBlock(nn.Module):
    """One cross-attention layer with residual + FFN."""
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, key, value, need_weights: bool = False):
        attn_out, attn_w = self.attn(query, key, value,
                                     need_weights=need_weights,
                                     average_attn_weights=True)
        query = self.norm1(query + attn_out)
        query = self.norm2(query + self.ff(query))
        return query, attn_w


# -- Main model -----------------------------------------------------------------

class BscanRegression(nn.Module):

    def __init__(
        self,
        junction_bps: int = 100,

        # -- Branch A: CNN -------------------------------------------------
        use_cnn: bool = True,
        cnn_channels: tuple = (256, 128),
        cnn_kernels:  tuple = (12, 8),
        cnn_pool:     int   = 4,

        # -- Branch B: Motif -----------------------------------------------
        use_motif: bool = False, # Disabled for bscan config
        n_motifs:  int  = 128,
        motif_width: int = 8,

        # -- Branch C: Stem (WC base-pairing map) -------------------------
        use_stem: bool = True,
        stem_channels: tuple = (16, 32, 64),
        stem_adaptive: int   = 4,

        # -- Branch D: Cross-Attention -------------------------------------
        use_attn:   bool = True,
        d_model:    int  = 128,
        n_heads:    int  = 4,
        n_attn_layers: int = 2,

        # -- Shared --------------------------------------------------------
        dropout: float = 0.3,
        d_hidden: int  = 256,
    ):
        super().__init__()
        assert use_cnn or use_motif or use_stem or use_attn, \
            "At least one branch must be enabled."

        self.junction_bps   = junction_bps
        self.use_cnn        = use_cnn
        self.use_motif      = use_motif
        self.use_stem       = use_stem
        self.use_attn       = use_attn
        self.n_motifs       = n_motifs
        L = junction_bps

        total_d = 0   # accumulate classifier input dim

        # -- Branch A: CNN -------------------------------------------------
        if use_cnn:
            c1, c2 = cnn_channels
            k1, k2 = cnn_kernels
            self.cnn = nn.Sequential(
                nn.Conv1d(4, c1, k1, padding=k1 // 2),
                nn.BatchNorm1d(c1), nn.ReLU(inplace=True),
                nn.MaxPool1d(cnn_pool),
                nn.Conv1d(c1, c2, k2, padding=k2 // 2),
                nn.BatchNorm1d(c2), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(8),
            )
            d_cnn = c2 * 8
            total_d += d_cnn * 2

        # -- Branch B: Motif -----------------------------------------------
        if use_motif:
            self.motif_conv = nn.Conv1d(4, n_motifs, motif_width,
                                        padding=motif_width // 2, bias=False)
            self.motif_bn   = nn.BatchNorm1d(n_motifs)
            total_d += n_motifs * 4

        # -- Branch C: Stem ------------------------------------------------
        if use_stem:
            c1s, c2s, c3s = stem_channels
            self.stem_cnn = nn.Sequential(
                nn.Conv2d(1, c1s, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(c1s, c2s, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(c2s, c3s, 3, padding=1), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(stem_adaptive),
            )
            d_stem   = c3s * stem_adaptive * stem_adaptive
            total_d += d_stem + 2 * L

        # -- Branch D: Cross-Attention -------------------------------------
        if use_attn:
            self.attn_proj = nn.Sequential(
                nn.Linear(4, d_model), nn.GELU()
            )
            self.attn_layers = nn.ModuleList([
                _CrossAttnBlock(d_model, n_heads, dropout)
                for _ in range(n_attn_layers)
            ])
            total_d += d_model

        # -- Classifier ----------------------------------------------------
        self.drop = nn.Dropout(dropout)
        # *** KEY CHANGE FOR REGRESSION: n_classes=1 ***
        self.classifier = Classifier(d_in=total_d, d_hiddens=[d_hidden],
                                     dropout=dropout, n_classes=1)

    def _cnn_feat(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x).view(x.size(0), -1)

    def _motif_scores(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.motif_bn(self.motif_conv(x))).max(dim=-1).values

    def forward(
        self,
        upper_seq: torch.Tensor,
        lower_seq: torch.Tensor,
        *, 
        return_aux: bool = False,
    ):
        L = self.junction_bps
        B = upper_seq.size(0)

        u_int = upper_seq[:, :, :L]
        u_exon= upper_seq[:, :, L:]
        l_exon= lower_seq[:, :, :L]
        l_int = lower_seq[:, :, L:]
        l_int_rc = _rc(l_int)

        parts = []
        aux   = {}

        if self.use_cnn:
            cnn_u = self.drop(self._cnn_feat(upper_seq))
            cnn_l = self.drop(self._cnn_feat(lower_seq))
            parts += [cnn_u, cnn_l]

        if self.use_motif:
            ui = self._motif_scores(u_int)
            ue = self._motif_scores(u_exon)
            le = self._motif_scores(l_exon)
            li = self._motif_scores(l_int)
            motif_feat = self.drop(torch.cat([ui, ue, le, li], dim=1))
            parts.append(motif_feat)

        if self.use_stem:
            bp = _bp_map(u_int, l_int_rc)
            row_max = bp.max(dim=-1).values
            col_max = bp.max(dim=-2).values
            x_stem  = self.stem_cnn(bp.unsqueeze(1))
            stem_feat = self.drop(x_stem.view(B, -1))
            parts += [stem_feat, row_max, col_max]

        if self.use_attn:
            u_emb = self.attn_proj(upper_seq.permute(0, 2, 1))
            l_rc  = _rc(lower_seq)
            l_rc_emb = self.attn_proj(l_rc.permute(0, 2, 1))

            for i, layer in enumerate(self.attn_layers):
                u_emb, _ = layer(u_emb, l_rc_emb, l_rc_emb)

            attn_feat = self.drop(u_emb.mean(dim=1))
            parts.append(attn_feat)
        
        combined = torch.cat(parts, dim=1)
        # The output is a single continuous value per sample
        output = self.classifier(combined)

        # For regression, we expect [B, 1], so we squeeze the last dimension
        if not return_aux:
            return output.squeeze(-1)
        return output.squeeze(-1), aux

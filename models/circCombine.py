"""
CircCombine: Modular combination of all interpretable branches.

Motivation
──────────
circcnnatt (AUC 0.9054) is our best model but:
  - Uses pretrained RNAErnie → forced batch_size=8 → 5× slower than one-hot models
  - CNN + attention, but no explicit motif / stem interpretability

CircCombine merges the best of all branches in a single one-hot model:

  Branch A — CNN (local features, fast)
    Shared two-layer 1D CNN on upper + lower sequences.
    Captures k-mer composition and splice-site signals.

  Branch B — Motif (interpretable, explicit PWMs)
    K learnable 1D filters (width W) on one-hot → global max-pool.
    Applied to all 4 regions: upper_intron, upper_exon, lower_exon, lower_intron.
    After training: visualise as PWMs, compare to ATtRACT/RBPDB.

  Branch C — Stem (structural, WC base-pairing)
    Hard Watson-Crick map: upper_intron ⊗ lower_intron_rc → [B, L, L].
    Processed by 2D CNN → compact stem features + row/col attribution.

  Branch D — Cross-Attention (global upper↔lower interaction)
    Lightweight one-hot projection (Linear(4→d)) + N cross-attention layers.
    query=upper, key=lower_rc, value=lower_rc — mirrors circcnnatt
    but without the pretrained bottleneck.

Each branch is independently enabled/disabled → ablation studies.

Speed comparison (seed=42, bs=128)
───────────────────────────────────
  circcnn       173s   AUC 0.8943   (CNN only)
  circmotif     523s   AUC 0.8831   (Motif + Stem)
  circstemv2    423s   AUC 0.8963   (CNN + Attn + Stem)
  circcnnatt    881s   AUC 0.9054   (CNN + Attn, pretrained)
  CircCombine  ~400s?  ???          (CNN + Motif + Stem + Attn, one-hot)

Input format (double one-hot, same as circcnn/circstem/circbialign/circmotif)
────────────────────────────────────────────────────────────────────────────
  upper_seq : [B, 4, 2L]  upper_intron ([:L]) + upper_exon ([L:])
  lower_seq : [B, 4, 2L]  lower_exon  ([:L]) + lower_intron ([L:])

Visualisation (return_aux=True)
────────────────────────────────
  motif_weights    [K, 4, W]     filter PWMs (Branch B)
  region_scores    dict          per-region motif scores (Branch B)
  bp_map           [B, L, L]     WC stem map (Branch C)
  row_max          [B, L]        upstream pairing profile (Branch C)
  col_max          [B, L]        downstream pairing profile (Branch C)
  attn_map         [B, 2L, 2L]   full cross-attention weights (Branch D)
  cnn_upper        [B, d_cnn]    CNN features — upper (Branch A)
  cnn_lower        [B, d_cnn]    CNN features — lower (Branch A)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import Classifier

_RC_PERM = [3, 2, 1, 0]   # A=0,C=1,G=2,T=3  →  T,G,C,A


def _rc(x: torch.Tensor) -> torch.Tensor:
    return x[:, _RC_PERM, :].flip(dims=[2])


def _bp_map(u_oh: torch.Tensor, l_rc_oh: torch.Tensor) -> torch.Tensor:
    """Hard WC base-pairing: [B, L, 4] @ [B, 4, L] → [B, L, L]"""
    return torch.bmm(u_oh.transpose(1, 2), l_rc_oh)


# ── Sub-modules ───────────────────────────────────────────────────────────────

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


# ── Main model ─────────────────────────────────────────────────────────────────

class CircCombine(nn.Module):

    def __init__(
        self,
        junction_bps: int = 100,

        # ── Branch A: CNN ─────────────────────────────────────────────────
        use_cnn: bool = True,
        cnn_channels: tuple = (256, 128),
        cnn_kernels:  tuple = (12, 8),
        cnn_pool:     int   = 4,
        cnn_style:    str   = "compact",

        # ── Branch B: Motif ───────────────────────────────────────────────
        use_motif: bool = True,
        n_motifs:  int  = 128,
        motif_width: int = 8,

        # ── Branch C: Stem (WC base-pairing map) ─────────────────────────
        use_stem: bool = True,
        stem_channels: tuple = (16, 32, 64),
        stem_adaptive: int   = 4,

        # ── Branch D: Cross-Attention ─────────────────────────────────────
        use_attn:   bool = True,
        d_model:    int  = 128,
        n_heads:    int  = 4,
        n_attn_layers: int = 2,

        # ── Shared ────────────────────────────────────────────────────────
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
        self.cnn_style      = cnn_style
        L = junction_bps

        # All branches must be defined BEFORE the dynamic dimension calculation
        # ── Branch A: CNN ─────────────────────────────────────────────────
        if use_cnn:
            c1, c2 = cnn_channels
            k1, k2 = cnn_kernels
            if cnn_style == "compact":
                self.cnn = nn.Sequential(
                    nn.Conv1d(4, c1, k1, padding=k1 // 2),
                    nn.BatchNorm1d(c1), nn.ReLU(inplace=True),
                    nn.MaxPool1d(cnn_pool),
                    nn.Conv1d(c1, c2, k2, padding=k2 // 2),
                    nn.BatchNorm1d(c2), nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool1d(8),
                )
            elif cnn_style == "circcnn":
                self.upper_cnn = nn.Sequential(
                    nn.Conv1d(4, c1, k1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(c1, c2, k2, stride=2, padding=k2 // 2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=5, stride=5),
                )
                self.lower_cnn = nn.Sequential(
                    nn.Conv1d(4, c1, k1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(c1, c2, k2, stride=2, padding=k2 // 2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=5, stride=5),
                )
            else:
                raise ValueError(f"Unsupported cnn_style: {cnn_style}")

        # ── Branch B: Motif ───────────────────────────────────────────────
        if use_motif:
            self.motif_conv = nn.Conv1d(4, n_motifs, motif_width,
                                        padding=motif_width // 2, bias=False)
            self.motif_bn   = nn.BatchNorm1d(n_motifs)

        # ── Branch C: Stem ────────────────────────────────────────────────
        if use_stem:
            c1s, c2s, c3s = stem_channels
            self.stem_cnn = nn.Sequential(
                nn.Conv2d(1, c1s, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(c1s, c2s, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(c2s, c3s, 3, padding=1), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(stem_adaptive),
            )

        # ── Branch D: Cross-Attention ─────────────────────────────────────
        if use_attn:
            self.attn_proj = nn.Sequential(
                nn.Linear(4, d_model), nn.GELU()
            )
            self.attn_layers = nn.ModuleList([
                _CrossAttnBlock(d_model, n_heads, dropout)
                for _ in range(n_attn_layers)
            ])

        # ── Dynamically determine classifier input dimension ──────────────────
        total_d = 0
        with torch.no_grad():
            dummy_upper = torch.randn(1, 4, 2 * L)
            dummy_lower = torch.randn(1, 4, 2 * L)

            if use_cnn:
                dummy_cnn_u = self._cnn_feat(dummy_upper, branch="upper")
                dummy_cnn_l = self._cnn_feat(dummy_lower, branch="lower")
                total_d += dummy_cnn_u.view(1, -1).size(1)
                total_d += dummy_cnn_l.view(1, -1).size(1)

            if use_motif:
                total_d += n_motifs * 4

            if use_stem:
                dummy_u_int = dummy_upper[:, :, :L]
                dummy_l_int_rc = _rc(dummy_lower[:, :, L:])
                dummy_bp_map = _bp_map(dummy_u_int, dummy_l_int_rc)
                dummy_stem_out = self.stem_cnn(dummy_bp_map.unsqueeze(1))
                d_stem = dummy_stem_out.view(1, -1).size(1)
                total_d += d_stem + 2 * L

            if use_attn:
                dummy_attn_proj = self.attn_proj(dummy_upper.permute(0, 2, 1))
                total_d += dummy_attn_proj.size(-1)

        # ── Classifier ────────────────────────────────────────────────────────
        self.drop = nn.Dropout(dropout)
        self.classifier = Classifier(d_in=total_d, d_hiddens=[d_hidden],
                                     dropout=dropout)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _cnn_feat(self, x: torch.Tensor, branch: str = "upper") -> torch.Tensor:
        """x: [B, 4, L_any] → [B, d_cnn]"""
        if self.cnn_style == "compact":
            y = self.cnn(x)
        elif branch == "upper":
            y = self.upper_cnn(x)
        elif branch == "lower":
            y = self.lower_cnn(x)
        else:
            raise ValueError(f"Unsupported CNN branch: {branch}")
        return y.view(x.size(0), -1)

    def _motif_scores(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 4, L] → [B, K]  global-max motif presence"""
        return F.relu(self.motif_bn(self.motif_conv(x))).max(dim=-1).values

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        upper_seq: torch.Tensor,
        lower_seq: torch.Tensor,
        *,
        return_aux: bool = False,
    ):
        L = self.junction_bps
        B = upper_seq.size(0)

        # Split sequences into 4 regions
        u_int = upper_seq[:, :, :L]     # upper intron  [B, 4, L]
        u_exon= upper_seq[:, :, L:]     # upper exon    [B, 4, L]
        l_exon= lower_seq[:, :, :L]     # lower exon    [B, 4, L]
        l_int = lower_seq[:, :, L:]     # lower intron  [B, 4, L]
        l_int_rc = _rc(l_int)           # RC of lower intron

        parts = []
        aux   = {}

        # ── Branch A: CNN ─────────────────────────────────────────────────
        if self.use_cnn:
            # Use full 2L sequences for maximum context
            cnn_u = self.drop(self._cnn_feat(upper_seq, branch="upper"))   # [B, d_cnn]
            cnn_l = self.drop(self._cnn_feat(lower_seq, branch="lower"))   # [B, d_cnn]
            parts += [cnn_u, cnn_l]
            if return_aux:
                aux['cnn_upper'] = cnn_u.detach()
                aux['cnn_lower'] = cnn_l.detach()

        # ── Branch B: Motif ───────────────────────────────────────────────
        if self.use_motif:
            ui = self._motif_scores(u_int)    # [B, K]
            ue = self._motif_scores(u_exon)
            le = self._motif_scores(l_exon)
            li = self._motif_scores(l_int)
            motif_feat = self.drop(torch.cat([ui, ue, le, li], dim=1))   # [B, 4K]
            parts.append(motif_feat)
            if return_aux:
                aux['motif_weights'] = self.motif_conv.weight.detach()
                aux['region_scores'] = {
                    'upper_intron': ui.detach(),
                    'upper_exon':   ue.detach(),
                    'lower_exon':   le.detach(),
                    'lower_intron': li.detach(),
                }

        # ── Branch C: Stem ────────────────────────────────────────────────
        if self.use_stem:
            bp = _bp_map(u_int, l_int_rc)                      # [B, L, L]
            row_max = bp.max(dim=-1).values                    # [B, L]
            col_max = bp.max(dim=-2).values                    # [B, L]
            x_stem  = self.stem_cnn(bp.unsqueeze(1))           # [B, C, s, s]
            stem_feat = self.drop(x_stem.view(B, -1))          # [B, d_stem]
            parts += [stem_feat, row_max, col_max]
            if return_aux:
                aux['bp_map']   = bp.detach()
                aux['row_max']  = row_max.detach()
                aux['col_max']  = col_max.detach()
                aux['stem_feat']= stem_feat.detach()

        # ── Branch D: Cross-Attention ─────────────────────────────────────
        if self.use_attn:
            # One-hot [B, 4, 2L] → [B, 2L, d_model]
            u_emb = self.attn_proj(upper_seq.permute(0, 2, 1))       # [B, 2L, d]
            l_rc  = _rc(lower_seq)                                    # [B, 4, 2L]
            l_rc_emb = self.attn_proj(l_rc.permute(0, 2, 1))         # [B, 2L, d]

            attn_w = None
            for i, layer in enumerate(self.attn_layers):
                need = return_aux and (i == len(self.attn_layers) - 1)
                u_emb, attn_w = layer(u_emb, l_rc_emb, l_rc_emb,
                                      need_weights=need)

            attn_feat = self.drop(u_emb.mean(dim=1))   # [B, d_model]
            parts.append(attn_feat)
            if return_aux and attn_w is not None:
                aux['attn_map'] = attn_w.detach()       # [B, 2L, 2L]

        # ── Fusion + Classification ───────────────────────────────────────
        combined = torch.cat(parts, dim=1)
        logits   = self.classifier(combined)

        if not return_aux:
            return logits
        return logits, aux

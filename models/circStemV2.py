"""
CircStemV2: Full-sequence attention (power) + intron stem map (interpretability)

Design rationale
────────────────
CircCNNATT (AUC 0.9065) beats CircCNN (AUC 0.8989) by adding multi-layer
cross-attention between upper and lower_rc.  CircStem (AUC 0.8867) adds
a biologically motivated intron/exon split but loses the full-sequence signal.

CircStemV2 combines both:

  Branch A — CNN features (like CircCNN/CircCNNATT)
    Multi-scale local motifs from upper + lower via shared 1D CNN.

  Branch B — Full cross-attention (like CircCNNATT, but scratch + one-hot)
    upper_seq (2L) cross-attends to RC(lower_seq) (2L) through N stacked
    attention layers.  Mean-pooled output captures global alignment.

  Branch C — Stem cross-attention (CircStem's biological path)
    Only the intron halves (first L tokens of each sequence) cross-attend.
    The resulting L×L map is processed by 2D CNN → stem_feat.
    stem_feat gates the exon junction features, mimicking the two-step
    backsplicing mechanism (intron pairing → spliceosome recognition).

Input format (double one-hot, same as circcnn/circstem, no tokenizer needed)
────────────
  upper_seq: [B, 4, 2L]   upper_intron ([:L]) + upper_exon ([L:])
  lower_seq: [B, 4, 2L]   lower_exon ([:L]) + lower_intron ([L:])
RC of lower_seq is computed internally.

Visualisation (return_aux=True)
────────────────────────────────
  stem_attn_map  [B, L, L]     intron pairing heatmap  ← primary
  gate           [B, d_junc]   stem→junction gate values
  full_attn_out  [B, 2L, d]    full-sequence attended embeddings
  stem_feat      [B, d_stem]   stem summary (UMAP, clustering)
  u_intron_emb   [B, L, d]     upper intron per-position embeddings
  l_intron_emb   [B, L, d]     lower intron RC per-position embeddings
"""

import torch
import torch.nn as nn

from .classifier import Classifier

# RC permutation: A=0,C=1,G=2,T=3  →  T=3,G=2,C=1,A=0
_RC_PERM = [3, 2, 1, 0]


def _rc(x: torch.Tensor) -> torch.Tensor:
    """Reverse complement of a one-hot sequence.
    Args: x [B, 4, L]
    Returns: [B, 4, L]
    """
    return x[:, _RC_PERM, :].flip(dims=[2])


class _CrossAttnLayer(nn.Module):
    """Single cross-attention layer with residual FFN (batch_first)."""
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, key, value):
        # cross-attention + residual
        attn_out, _ = self.attn(query, key, value)
        query = self.norm1(query + attn_out)
        # FFN + residual
        query = self.norm2(query + self.ff(query))
        return query


class CircStemV2(nn.Module):

    def __init__(
        self,
        junction_bps: int = 100,
        d_model: int = 128,
        n_heads: int = 4,
        n_attn_layers: int = 3,       # stacked cross-attention layers (Branch B)
        dropout: float = 0.1,
        # CNN encoder (Branch A)
        cnn_channels: int = 256,
        cnn_kernel: int = 11,
        cnn_adaptive_len: int = 8,    # AdaptiveAvgPool1d output length
        # 2D CNN on stem map (Branch C)
        stem_conv_ch: tuple = (16, 32, 64),
        stem_adaptive_size: int = 4,  # AdaptiveAvgPool2d output side
    ):
        super().__init__()
        self.junction_bps = junction_bps
        L = junction_bps

        # ── Shared one-hot → d_model embedding ──────────────────────────
        # Used by both full and stem attention branches.
        self.embed = nn.Conv1d(4, d_model, kernel_size=1)   # [B, d, seq_len]

        # ── Branch A: shared CNN encoder ────────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv1d(4, cnn_channels, cnn_kernel, padding=cnn_kernel // 2),
            nn.BatchNorm1d(cnn_channels), nn.ReLU(inplace=True), nn.MaxPool1d(4),
            nn.Conv1d(cnn_channels, cnn_channels, cnn_kernel, padding=cnn_kernel // 2),
            nn.BatchNorm1d(cnn_channels), nn.ReLU(inplace=True),
        )
        self.cnn_pool = nn.AdaptiveAvgPool1d(cnn_adaptive_len)
        self.d_cnn = cnn_channels * cnn_adaptive_len         # per sequence

        # ── Branch B: full-sequence multi-layer cross-attention ─────────
        # query=upper_emb [B, 2L, d], key/value=lower_rc_emb [B, 2L, d]
        self.full_attn_layers = nn.ModuleList([
            _CrossAttnLayer(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_attn_layers)
        ])
        self.d_full = d_model                               # mean-pooled

        # ── Branch C: intron stem cross-attention + 2D CNN ───────────────
        self.stem_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        c1, c2, c3 = stem_conv_ch
        self.stem_map_cnn = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.stem_pool = nn.AdaptiveAvgPool2d(stem_adaptive_size)
        self.d_stem = c3 * stem_adaptive_size * stem_adaptive_size  # 64*16 = 1024

        # ── Gate: stem modulates junction (exon) features ───────────────
        self.d_junc = d_model * 2
        self.gate_proj = nn.Sequential(
            nn.Linear(self.d_stem, self.d_junc),
            nn.Sigmoid(),
        )

        # ── Classifier ───────────────────────────────────────────────────
        total_dim = self.d_cnn * 2 + self.d_full + self.d_stem + self.d_junc
        self.classifier = Classifier(d_in=total_dim)
        self.dropout = nn.Dropout(dropout)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _embed(self, onehot: torch.Tensor) -> torch.Tensor:
        """[B, 4, L] → [B, L, d_model]"""
        return self.embed(onehot).transpose(1, 2)

    def _cnn_feat(self, onehot: torch.Tensor) -> torch.Tensor:
        """[B, 4, L] → [B, d_cnn] (flattened)"""
        x = self.cnn(onehot)                  # [B, C, L']
        x = self.cnn_pool(x)                  # [B, C, adaptive_len]
        return x.view(x.size(0), -1)          # [B, d_cnn]

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        upper_seq: torch.Tensor,
        lower_seq: torch.Tensor,
        *,
        return_aux: bool = False,
    ):
        """
        Args:
            upper_seq: [B, 4, 2L]  one-hot (upper_intron | upper_exon)
            lower_seq: [B, 4, 2L]  one-hot (lower_exon  | lower_intron)
            return_aux: if True, return dict of intermediate tensors
        Returns:
            logits [B, 2], and optionally aux dict.
        """
        L = self.junction_bps

        # RC of lower_seq computed internally
        lower_rc = _rc(lower_seq)                            # [B, 4, 2L]

        # ── Branch A: CNN features ────────────────────────────────────
        u_cnn  = self._cnn_feat(upper_seq)                   # [B, d_cnn]
        l_cnn  = self._cnn_feat(lower_seq)                   # [B, d_cnn]

        # ── Branch B: full cross-attention ───────────────────────────
        u_emb     = self._embed(upper_seq)                   # [B, 2L, d]
        l_rc_emb  = self._embed(lower_rc)                    # [B, 2L, d]

        # Stacked cross-attention: upper queries lower_rc
        full_out = u_emb
        for layer in self.full_attn_layers:
            full_out = layer(full_out, l_rc_emb, l_rc_emb)  # [B, 2L, d]
        full_feat = self.dropout(full_out.mean(dim=1))       # [B, d_model]

        # ── Branch C: stem cross-attention (intron only) ─────────────
        u_intron   = u_emb[:, :L, :]                         # [B, L, d]
        l_intron_rc = l_rc_emb[:, :L, :]                     # [B, L, d]

        _, stem_attn_map = self.stem_attn(
            query=u_intron, key=l_intron_rc, value=l_intron_rc,
            need_weights=True, average_attn_weights=True,    # [B, L, L]
        )
        x_stem = self.stem_map_cnn(stem_attn_map.unsqueeze(1))
        x_stem = self.stem_pool(x_stem)
        stem_feat = self.dropout(x_stem.view(x_stem.size(0), -1))  # [B, d_stem]

        # ── Gate: exon junction features modulated by stem ───────────
        u_exon_feat  = u_emb[:, L:, :].mean(dim=1)          # [B, d]
        l_exon_feat  = l_rc_emb[:, L:, :].mean(dim=1)       # [B, d]  (RC of lower_exon)
        junc_feat    = torch.cat([u_exon_feat, l_exon_feat], dim=1)  # [B, d_junc]

        gate       = self.gate_proj(stem_feat)               # [B, d_junc], (0,1)
        junc_gated = junc_feat * gate                        # [B, d_junc]

        # ── Fusion & Classification ───────────────────────────────────
        combined = torch.cat([u_cnn, l_cnn, full_feat, stem_feat, junc_gated], dim=1)
        logits = self.classifier(combined)

        if not return_aux:
            return logits

        return logits, {
            # ── Primary visualisation ──
            "stem_attn_map": stem_attn_map.detach(),  # [B, L, L]  intron pairing heatmap
            "gate":          gate.detach(),            # [B, d_junc] junction gate values
            # ── Full-sequence attention ──
            "full_attn_out": full_out.detach(),        # [B, 2L, d] per-position attended
            # ── Embedding spaces ──
            "stem_feat":     stem_feat.detach(),       # [B, d_stem] UMAP / clustering
            # ── Per-position intron representations ──
            "u_intron_emb":  u_intron.detach(),        # [B, L, d]
            "l_intron_emb":  l_intron_rc.detach(),     # [B, L, d]
        }

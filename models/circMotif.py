"""
CircMotif: Interpretable Motif-Discovery Model for circRNA Back-Splicing Prediction

Biological motivation
─────────────────────
Back-splicing is driven by two complementary mechanisms:
  1. RNA-binding protein (RBP) binding sites (motifs) in the flanking sequences
  2. Reverse-complementary stem-loop structures in the flanking introns

This model makes both mechanisms explicitly learnable and inspectable:

  A. Motif Branch
     K learnable 1D convolutional filters (width W) on one-hot sequences.
     Each filter acts as a position weight matrix (PWM) — exactly a motif detector.
     Shared across 4 sequence regions:
       upper_intron | upper_exon | lower_exon | lower_intron
     Global max-pooling over positions → K motif presence scores per region.
     After training: visualize each filter as a PWM and match against
     RBP databases (ATtRACT, RBPDB, ENCODE eCLIP peaks).

  B. Stem Branch
     Hard Watson-Crick base-pairing map computed directly from one-hot:
       bp[i,j] = 1 iff upper_intron[i] and lower_intron_rc[j] share
                    the same nucleotide (after RC, complementary positions match).
     row_max / col_max over the [L×L] map → per-position pairing strength [2L].
     Optional: small 2D CNN extracts local stem-pattern features.

  C. Classifier
     MLP on [motif_scores (4K) | stem_features (2L + d_stem)].
     Gradient × input (or SHAP) gives per-motif attribution.

Input format (double one-hot, same as circcnn / circstem / circbialign)
────────────────────────────────────────────────────────────────────────
  upper_seq : [B, 4, 2L]  upper_intron ([:L]) + upper_exon ([L:])
  lower_seq : [B, 4, 2L]  lower_exon  ([:L]) + lower_intron ([L:])

Visualisation (return_aux=True)
────────────────────────────────
  motif_weights      [K, 4, W]   raw filter weights (→ convert to PWM)
  region_scores      dict        {region_name: [B, K]} per-region motif presence
  bp_map             [B, L, L]   Watson-Crick base-pairing map (0/1)
  row_max            [B, L]      upstream intron best-pairing profile
  col_max            [B, L]      downstream intron best-pairing profile
  stem_feat          [B, d_stem] 2D-CNN stem features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import Classifier

_RC_PERM = [3, 2, 1, 0]   # A=0,C=1,G=2,T=3  →  T,G,C,A


def _rc(x: torch.Tensor) -> torch.Tensor:
    """Reverse complement of a one-hot sequence [B, 4, L] → [B, 4, L]"""
    return x[:, _RC_PERM, :].flip(dims=[2])


def _bp_map(upper_oh: torch.Tensor, lower_rc_oh: torch.Tensor) -> torch.Tensor:
    """Hard Watson-Crick base-pairing map from one-hot sequences.

    After taking RC of lower_intron, complementary pairs share the same
    nucleotide identity (A-T → A-A; G-C → G-G).
    bp[b,i,j] = 1 iff position i of upper_intron and position j of
                     lower_intron_rc are Watson-Crick complementary.

    Args:
        upper_oh:    [B, 4, L]  one-hot
        lower_rc_oh: [B, 4, L]  one-hot of RC(lower_intron)
    Returns:
        [B, L, L]  values in {0, 1}
    """
    u = upper_oh.transpose(1, 2)   # [B, L, 4]
    l = lower_rc_oh                # [B, 4, L]
    return torch.bmm(u, l)         # [B, L, L]


class CircMotif(nn.Module):

    def __init__(
        self,
        junction_bps: int = 100,
        # Motif branch
        n_motifs: int = 256,
        motif_width: int = 8,
        # Stem branch (2D CNN on bp_map)
        stem_conv_channels: tuple = (16, 32, 64),
        stem_adaptive: int = 4,
        # Classifier
        dropout: float = 0.3,
        d_hidden: int = 256,
    ):
        super().__init__()
        self.junction_bps = junction_bps
        self.n_motifs = n_motifs
        self.motif_width = motif_width
        L = junction_bps

        # ── Motif Branch ─────────────────────────────────────────────────
        # Single set of K filters shared across all 4 sequence regions.
        # No bias: keeps the filter interpretable as a pure sequence scorer.
        # padding=motif_width//2 keeps output length ≈ input length.
        self.motif_conv = nn.Conv1d(
            in_channels=4,
            out_channels=n_motifs,
            kernel_size=motif_width,
            padding=motif_width // 2,
            bias=False,
        )
        self.motif_bn = nn.BatchNorm1d(n_motifs)

        # ── Stem Branch ───────────────────────────────────────────────────
        # 2D CNN on the L×L WC base-pairing map
        c1, c2, c3 = stem_conv_channels
        self.stem_cnn = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(stem_adaptive),
        )
        d_stem = c3 * stem_adaptive * stem_adaptive   # e.g. 64*4*4 = 1024

        # ── Classifier ────────────────────────────────────────────────────
        # Input = 4 regions × K motif scores + row_max (L) + col_max (L) + stem_feat
        d_in = 4 * n_motifs + 2 * L + d_stem
        self.classifier = Classifier(d_in=d_in, d_hiddens=[d_hidden], dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _motif_scores(self, onehot: torch.Tensor) -> torch.Tensor:
        """Apply motif filters and global-max-pool over positions.

        Args:
            onehot: [B, 4, L]
        Returns:
            scores: [B, K]  — max activation of each motif across the sequence
        """
        x = self.motif_conv(onehot)      # [B, K, L']
        x = self.motif_bn(x)
        x = F.relu(x)
        return x.max(dim=-1).values      # [B, K]  global max-pool

    # ── Forward ───────────────────────────────────────────────────────────

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
            return_aux: if True, also return interpretability dict

        Returns:
            logits [B, 2], and optionally aux dict
        """
        L = self.junction_bps

        # ── Split sequences ───────────────────────────────────────────────
        upper_intron = upper_seq[:, :, :L]       # [B, 4, L]
        upper_exon   = upper_seq[:, :, L:]       # [B, 4, L]
        lower_exon   = lower_seq[:, :, :L]       # [B, 4, L]
        lower_intron = lower_seq[:, :, L:]       # [B, 4, L]
        lower_intron_rc = _rc(lower_intron)      # [B, 4, L]

        # ── A: Motif Branch ───────────────────────────────────────────────
        # Apply same K filters to all 4 regions
        ui_scores = self._motif_scores(upper_intron)     # [B, K]
        ue_scores = self._motif_scores(upper_exon)       # [B, K]
        le_scores = self._motif_scores(lower_exon)       # [B, K]
        li_scores = self._motif_scores(lower_intron)     # [B, K]  (forward strand)

        motif_feat = self.dropout(
            torch.cat([ui_scores, ue_scores, le_scores, li_scores], dim=1)
        )   # [B, 4K]

        # ── B: Stem Branch ────────────────────────────────────────────────
        bp = _bp_map(upper_intron, lower_intron_rc)     # [B, L, L]

        # Per-position pairing strength (interpretable)
        row_max = bp.max(dim=-1).values                  # [B, L]
        col_max = bp.max(dim=-2).values                  # [B, L]

        # 2D CNN on stem map
        x_stem = self.stem_cnn(bp.unsqueeze(1))          # [B, C, s, s]
        stem_feat = self.dropout(x_stem.view(x_stem.size(0), -1))  # [B, d_stem]

        # ── Fusion & Classification ───────────────────────────────────────
        combined = torch.cat(
            [motif_feat, row_max, col_max, stem_feat], dim=1
        )   # [B, 4K + 2L + d_stem]
        logits = self.classifier(combined)

        if not return_aux:
            return logits

        return logits, {
            # Motif filter weights (visualize as PWMs post-training)
            "motif_weights":  self.motif_conv.weight.detach(),   # [K, 4, W]
            # Per-region motif presence scores
            "region_scores": {
                "upper_intron": ui_scores.detach(),
                "upper_exon":   ue_scores.detach(),
                "lower_exon":   le_scores.detach(),
                "lower_intron": li_scores.detach(),
            },
            # Stem structure
            "bp_map":         bp.detach(),                       # [B, L, L]
            "row_max":        row_max.detach(),                  # [B, L]
            "col_max":        col_max.detach(),                  # [B, L]
            "stem_feat":      stem_feat.detach(),                # [B, d_stem]
        }

"""
CircSplice: Biologically-motivated architecture for circRNA back-splice prediction.

Designed based on discovered key signals:
  1. Poly-A signal (AATAAA) near downstream 5' splice site in lower_exon
  2. Weak 5'/3' splice sites → non-canonical GT/AG
  3. lower_exon is the most important sequence region (AUC drop 0.36 when removed)

Architecture: Three branches
  A. Global CNN      — full 200nt sequence context (upper/lower)
  B. Junction CNN    — focused on ±junction_win nt around each splice site
                       3'ss: upper_intron[-W:] + upper_exon[:W]   [4, 2W]
                       5'ss: lower_exon[-W:]  + lower_intron[:W]  [4, 2W]
  C. Splice features — explicit splice site strength + poly-A detection
                       (learnable PWM-style filters on the junction window)

Input (same as circcnn / circcombine):
  upper : [B, 4, 2L]  upper_intron[:L] + upper_exon[L:]
  lower : [B, 4, 2L]  lower_exon[:L]  + lower_intron[L:]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .classifier import Classifier


def _cnn_block(in_ch, out_ch, kernel, pool=2, dropout=0.2):
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel, padding=kernel//2),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(),
        nn.MaxPool1d(pool),
        nn.Dropout(dropout),
    )


class CircSplice(nn.Module):
    """
    Parameters
    ----------
    junction_bps : int
        Number of nt on each side of the junction (L). Input length = 2L.
    junction_win : int
        Window around each splice site for Junction CNN branch (default 30).
        Processes [upper_intron[-W:] + upper_exon[:W]] and [lower_exon[-W:] + lower_intron[:W]].
    d_cnn : int
        Channels in global CNN branch.
    d_junc : int
        Channels in junction CNN branch.
    n_pwm : int
        Number of PWM-style filters for splice feature detection.
    pwm_width : int
        Width of PWM filters (6 = hexamer, good for AATAAA detection).
    dropout : float
    """

    def __init__(self, junction_bps=100, junction_win=30,
                 d_cnn=128, d_junc=64, n_pwm=64, pwm_width=6,
                 dropout=0.3):
        super().__init__()
        self.L    = junction_bps
        self.W    = junction_win

        # ── Branch A: Global CNN ───────────────────────────────────────────
        # Input: [B, 4, 2L]
        self.global_cnn = nn.Sequential(
            _cnn_block(4,      d_cnn,     9, pool=2, dropout=dropout),
            _cnn_block(d_cnn,  d_cnn*2,   7, pool=2, dropout=dropout),
            _cnn_block(d_cnn*2,d_cnn*2,   5, pool=2, dropout=dropout),
            nn.AdaptiveMaxPool1d(1),
        )
        d_global = d_cnn * 2  # per stream (upper + lower)

        # ── Branch B: Junction CNN ─────────────────────────────────────────
        # Input: [B, 4, 2W]  — focused window around each splice site
        self.junc_cnn = nn.Sequential(
            _cnn_block(4,      d_junc,    5, pool=2, dropout=dropout),
            _cnn_block(d_junc, d_junc*2,  3, pool=2, dropout=dropout),
            nn.AdaptiveMaxPool1d(1),
        )
        d_junc_out = d_junc * 2  # per splice site (3'ss + 5'ss)

        # ── Branch C: PWM splice feature detectors ─────────────────────────
        # Two sets of PWM filters:
        #   c1) Junction PWM: applied to ±W window around each splice site
        #       → detect GT/AG dinucleotides, short splice site motifs
        #   c2) Full-exon PWM: applied to FULL lower_exon (100nt)
        #       → detect AATAAA (poly-A signal) anywhere in the exon
        #       This is key: AATAAA enrichment is BS=60.8% vs LS=11.3% anywhere in lower_exon
        self.pwm_conv    = nn.Conv1d(4, n_pwm,     pwm_width, padding=pwm_width//2, bias=False)
        self.pwm_conv_le = nn.Conv1d(4, n_pwm//2,  pwm_width, padding=pwm_width//2, bias=False)
        self.pwm_conv_ui = nn.Conv1d(4, n_pwm//2,  pwm_width, padding=pwm_width//2, bias=False)
        d_pwm_junc = n_pwm        # junction PWM (×2 splice sites)
        d_pwm_exon = n_pwm // 2   # full-exon PWM (lower_exon + upper_intron)

        # ── Classifier ────────────────────────────────────────────────────
        # global(upper+lower) + junc_cnn(3ss+5ss) + pwm_junc(3ss+5ss) + pwm_exon(le+ui)
        d_total = (d_global * 2 + d_junc_out * 2
                   + d_pwm_junc * 2 + d_pwm_exon * 2)
        self.classifier = Classifier(d_total, d_hiddens=[256], dropout=dropout)

    def _extract_junction_windows(self, upper, lower):
        """
        Extract ±W nt around each splice site.

        upper : [B, 4, 2L]  — [upper_intron | upper_exon]
        lower : [B, 4, 2L]  — [lower_exon   | lower_intron]

        3'ss window:  upper_intron[-W:] + upper_exon[:W]
                      = upper[:, :, L-W : L+W]
        5'ss window:  lower_exon[-W:]  + lower_intron[:W]
                      = lower[:, :, L-W : L+W]
        """
        L, W = self.L, self.W
        # Clamp W to valid range
        w = min(W, L)
        ss3 = upper[:, :, L-w : L+w]   # [B, 4, 2W]
        ss5 = lower[:, :, L-w : L+w]   # [B, 4, 2W]
        return ss3, ss5

    def forward(self, upper, lower, return_aux=False):
        L = self.L

        # ── Branch A: Global CNN ───────────────────────────────────────────
        f_up = self.global_cnn(upper).squeeze(-1)   # [B, d_global]
        f_lo = self.global_cnn(lower).squeeze(-1)

        # ── Region extraction ──────────────────────────────────────────────
        # upper: [upper_intron[:L] | upper_exon[L:]]
        # lower: [lower_exon[:L]  | lower_intron[L:]]
        upper_intron = upper[:, :, :L]   # [B, 4, L]
        lower_exon   = lower[:, :, :L]   # [B, 4, L]  ← AATAAA signal lives here
        ss3, ss5     = self._extract_junction_windows(upper, lower)  # [B, 4, 2W]

        # ── Branch B: Junction CNN ─────────────────────────────────────────
        f_ss3_junc = self.junc_cnn(ss3).squeeze(-1)   # [B, d_junc*2]
        f_ss5_junc = self.junc_cnn(ss5).squeeze(-1)

        # ── Branch C1: Junction PWM (splice site motifs) ───────────────────
        f_ss3_pwm = F.relu(self.pwm_conv(ss3)).max(dim=-1).values   # [B, n_pwm]
        f_ss5_pwm = F.relu(self.pwm_conv(ss5)).max(dim=-1).values

        # ── Branch C2: Full-exon PWM (poly-A signal detection) ────────────
        # Apply PWM to full lower_exon (100nt) → catch AATAAA anywhere
        f_le_pwm = F.relu(self.pwm_conv_le(lower_exon)).max(dim=-1).values    # [B, n_pwm//2]
        # Apply PWM to upper_intron (branch point / PPT detection)
        f_ui_pwm = F.relu(self.pwm_conv_ui(upper_intron)).max(dim=-1).values  # [B, n_pwm//2]

        # ── Concatenate all features ───────────────────────────────────────
        feat = torch.cat([
            f_up, f_lo,                    # global context
            f_ss3_junc, f_ss5_junc,        # junction CNN
            f_ss3_pwm, f_ss5_pwm,          # junction PWM
            f_le_pwm, f_ui_pwm,            # full-region PWM
        ], dim=1)

        logits = self.classifier(feat)

        if not return_aux:
            return logits

        return logits, {
            'pwm_weights':    self.pwm_conv.weight.detach(),     # [n_pwm, 4, W]
            'pwm_le_weights': self.pwm_conv_le.weight.detach(),  # [n_pwm//2, 4, W]
            'pwm_ss3':        f_ss3_pwm.detach(),
            'pwm_ss5':        f_ss5_pwm.detach(),
            'pwm_le':         f_le_pwm.detach(),
            'pwm_ui':         f_ui_pwm.detach(),
            'ss3_window':     ss3.detach(),
            'ss5_window':     ss5.detach(),
        }

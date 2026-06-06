"""
CircBiAlign: Bilinear Neural Alignment for circRNA back-splicing prediction

Core idea
─────────
Standard cross-attention computes Q·K^T (dot-product similarity).
We replace this with a bilinear form:

    S[i,j] = u_intron[i] @ W @ l_intron_rc[j]

where W is a learned d×d "complementarity transform" matrix.

Additionally, a hard base-pairing prior is computed directly from the
one-hot sequences and added to S:

    S_total = S_bilinear/√d  +  α · bp_prior

bp_prior[i,j] = 1 if upper_intron[i] and lower_intron_rc[j] have the
                    same nucleotide (A↔A, C↔C, G↔G, T↔T after RC)
               = 0 otherwise

Because RC(lower_intron) makes complementary stem positions share the
same nucleotide identity (A-T pair becomes A-A after RC), this is
exactly the Watson-Crick base-pairing signal.

W is initialised as the identity matrix (bilinear = dot product at start),
then fine-tuned to learn higher-order contextual complementarity patterns.
α is a learnable scalar that balances neural vs physical alignment.

Contributions
─────────────
1. Performance  — bilinear has strictly more expressivity than dot-product
2. Interpretability
     • S_total  [B, L, L]  — which intron positions form the RNA stem
     • W        [d, d]     — what "complementarity" means in embedding space
     • α        scalar     — data-driven weight of base-pairing vs context
3. Biological insight
     • S can be directly compared with RCSFinder / IntaRNA / RNAfold outputs
     • W post-analysis reveals motifs beyond simple A-T / G-C pairing
     • Per-position alignment strength (row_max, col_max) highlights
       which intron regions drive the BS vs LS decision

Input format (double one-hot, same pipeline as circcnn/circstem)
───────────────────────────────────────────────────────────────────
  upper_seq: [B, 4, 2L]  upper_intron ([:L]) + upper_exon ([L:])
  lower_seq: [B, 4, 2L]  lower_exon  ([:L]) + lower_intron ([L:])
RC is computed internally.

Visualisation (return_aux=True)
────────────────────────────────
  S_total      [B, L, L]   full alignment map  (bilinear + bp prior)
  S_bilinear   [B, L, L]   purely neural component
  bp_prior     [B, L, L]   hard base-pairing component (0/1)
  row_max      [B, L]      per-upstream-position best alignment score
  col_max      [B, L]      per-downstream-position best alignment score
  alpha        scalar      learned prior weight
  align_feat   [B, d_ali]  2D-CNN features from S_total
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import Classifier

_RC_PERM = [3, 2, 1, 0]   # A=0,C=1,G=2,T=3  →  T,G,C,A


def _rc(x: torch.Tensor) -> torch.Tensor:
    """Reverse complement of a one-hot sequence [B, 4, L] → [B, 4, L]"""
    return x[:, _RC_PERM, :].flip(dims=[2])


class CircBiAlign(nn.Module):

    def __init__(
        self,
        junction_bps: int = 100,
        d_model: int = 128,
        dropout: float = 0.1,
        # Length-preserving intron encoder
        enc_channels: int = 128,
        enc_kernel: int = 7,
        enc_layers: int = 2,          # stacked Conv1d layers
        # 2D CNN on alignment map
        ali_conv_ch: tuple = (16, 32, 64),
        ali_adaptive: int = 4,        # AdaptiveAvgPool2d side
        # Junction encoder
        junc_adaptive: int = 8,       # AdaptiveAvgPool1d length
        junc_channels: int = 128,
    ):
        super().__init__()
        self.junction_bps = junction_bps
        self.d_model = d_model
        L = junction_bps

        # ── Shared length-preserving CNN encoder ─────────────────────────
        # No pooling: output length = input length (needed for position-wise alignment)
        layers = []
        in_ch = 4
        for _ in range(enc_layers):
            layers += [
                nn.Conv1d(in_ch, enc_channels, enc_kernel, padding=enc_kernel // 2),
                nn.BatchNorm1d(enc_channels),
                nn.ReLU(inplace=True),
            ]
            in_ch = enc_channels
        self.encoder = nn.Sequential(*layers)
        # Project to d_model if needed
        self.enc_proj = nn.Linear(enc_channels, d_model) \
            if enc_channels != d_model else nn.Identity()

        # ── Bilinear complementarity transform ───────────────────────────
        # W: learned d×d matrix.  Initialised as identity so that at t=0
        # the bilinear reduces to dot-product (cosine-like) similarity.
        self.W = nn.Parameter(torch.eye(d_model))

        # α: learnable weight on the hard base-pairing prior
        self.alpha = nn.Parameter(torch.ones(1))

        # ── 2D CNN on alignment map S [B, 1, L, L] ───────────────────────
        c1, c2, c3 = ali_conv_ch
        self.ali_cnn = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.ali_pool = nn.AdaptiveAvgPool2d(ali_adaptive)
        self.d_ali = c3 * ali_adaptive * ali_adaptive              # 64*16 = 1024

        # ── Junction encoder (separate, with pooling) ─────────────────────
        self.junc_cnn = nn.Sequential(
            nn.Conv1d(4, junc_channels, enc_kernel, padding=enc_kernel // 2),
            nn.BatchNorm1d(junc_channels), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(junc_channels, junc_channels, enc_kernel, padding=enc_kernel // 2),
            nn.BatchNorm1d(junc_channels), nn.ReLU(inplace=True),
        )
        self.junc_pool = nn.AdaptiveAvgPool1d(junc_adaptive)
        d_junc_single = junc_channels * junc_adaptive
        self.d_junc = d_junc_single * 2                             # upper + lower exon

        # ── Gate: alignment strength → junction scale ─────────────────────
        # row_max [B, L] + col_max [B, L] → gate for junction
        self.gate_proj = nn.Sequential(
            nn.Linear(2 * L, self.d_junc),
            nn.Sigmoid(),
        )

        # ── Classifier ────────────────────────────────────────────────────
        # Features: 2D-CNN alignment + row_max + col_max + gated junction
        total_dim = self.d_ali + 2 * L + self.d_junc
        self.classifier = Classifier(d_in=total_dim)
        self.dropout = nn.Dropout(dropout)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _encode(self, onehot: torch.Tensor) -> torch.Tensor:
        """Shared length-preserving encoder.
        Args: onehot [B, 4, L]
        Returns: [B, L, d_model]
        """
        x = self.encoder(onehot)                   # [B, C, L]
        return self.enc_proj(x.transpose(1, 2))    # [B, L, d_model]

    @staticmethod
    def _bp_score(upper_oh: torch.Tensor,
                  lower_rc_oh: torch.Tensor) -> torch.Tensor:
        """Hard base-pairing prior from one-hot sequences.

        After taking RC, complementary positions share the same nucleotide
        identity (A-T pair → A-A after RC; G-C pair → G-G after RC).
        So bp[i,j] = 1 iff same nucleotide at position i of upper_intron
        and position j of lower_intron_rc.

        Args:
            upper_oh:    [B, 4, L]  one-hot
            lower_rc_oh: [B, 4, L]  one-hot (RC of lower_intron)
        Returns:
            [B, L, L]  values in {0, 1}
        """
        # [B, L, 4] @ [B, 4, L] = [B, L, L]
        u = upper_oh.transpose(1, 2)                    # [B, L, 4]
        l = lower_rc_oh                                 # [B, 4, L]
        return torch.bmm(u, l)                          # [B, L, L]

    def _junc_feat(self, onehot: torch.Tensor) -> torch.Tensor:
        """Junction CNN encoder.
        Args: onehot [B, 4, L]
        Returns: [B, d_junc_single]
        """
        x = self.junc_cnn(onehot)                      # [B, C, L']
        x = self.junc_pool(x)                          # [B, C, junc_adaptive]
        return x.view(x.size(0), -1)                   # [B, d_junc_single]

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
            return_aux: if True, also return analysis dict
        Returns:
            logits [B, 2], and optionally aux dict
        """
        L = self.junction_bps

        # ── Split + RC ────────────────────────────────────────────────────
        upper_intron   = upper_seq[:, :, :L]         # [B, 4, L]
        upper_exon     = upper_seq[:, :, L:]         # [B, 4, L]
        lower_exon     = lower_seq[:, :, :L]         # [B, 4, L]
        lower_intron   = lower_seq[:, :, L:]         # [B, 4, L]
        lower_intron_rc = _rc(lower_intron)          # [B, 4, L]  RC of lower_intron

        # ── Contextual embeddings ─────────────────────────────────────────
        u_emb = self._encode(upper_intron)            # [B, L, d]
        l_emb = self._encode(lower_intron_rc)         # [B, L, d]  (shared encoder)

        # ── Bilinear alignment score ──────────────────────────────────────
        # S_bilinear[b,i,j] = u_emb[b,i] @ W @ l_emb[b,j]
        u_w = u_emb @ self.W                          # [B, L, d]  apply W to queries
        S_bilinear = torch.bmm(u_w, l_emb.transpose(1, 2)) / (self.d_model ** 0.5)
        # S_bilinear: [B, L, L]

        # ── Hard base-pairing prior ───────────────────────────────────────
        bp = self._bp_score(upper_intron, lower_intron_rc)  # [B, L, L]

        # ── Combined alignment map ────────────────────────────────────────
        S = S_bilinear + self.alpha * bp             # [B, L, L]

        # ── 2D CNN on alignment map ───────────────────────────────────────
        x_ali = self.ali_cnn(S.unsqueeze(1))         # [B, C, H, W]
        x_ali = self.ali_pool(x_ali)                 # [B, C, s, s]
        ali_feat = self.dropout(x_ali.view(x_ali.size(0), -1))  # [B, d_ali]

        # ── Per-position alignment features ──────────────────────────────
        row_max = S.max(dim=-1).values               # [B, L]  each upstream best
        col_max = S.max(dim=-2).values               # [B, L]  each downstream best

        # ── Junction features (gated by alignment) ────────────────────────
        ue_feat = self._junc_feat(upper_exon)        # [B, d/2]
        le_feat = self._junc_feat(lower_exon)        # [B, d/2]
        junc_feat = torch.cat([ue_feat, le_feat], dim=1)  # [B, d_junc]

        rowcol = torch.cat([row_max, col_max], dim=1)    # [B, 2L]
        gate   = self.gate_proj(rowcol)                  # [B, d_junc], (0,1)
        junc_gated = junc_feat * gate                    # [B, d_junc]

        # ── Fusion & Classification ───────────────────────────────────────
        combined = torch.cat([ali_feat, rowcol, junc_gated], dim=1)
        logits = self.classifier(combined)

        if not return_aux:
            return logits

        return logits, {
            # ── Primary: alignment maps ──
            "S_total":     S.detach(),                # [B, L, L]  full alignment
            "S_bilinear":  S_bilinear.detach(),       # [B, L, L]  neural component
            "bp_prior":    bp.detach(),               # [B, L, L]  hard prior (0/1)
            # ── Per-position attribution ──
            "row_max":     row_max.detach(),          # [B, L]  upstream position strength
            "col_max":     col_max.detach(),          # [B, L]  downstream position strength
            # ── Learned parameters ──
            "alpha":       self.alpha.detach(),       # scalar: learned prior weight
            "W":           self.W.detach(),           # [d, d] complementarity transform
            # ── Embedding spaces ──
            "align_feat":  ali_feat.detach(),         # [B, d_ali]  for UMAP
            "u_intron_emb": u_emb.detach(),           # [B, L, d]
            "l_intron_emb": l_emb.detach(),           # [B, L, d]
        }

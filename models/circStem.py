import torch
import torch.nn as nn

from .mamba2 import Mamba2, Mamba2Config, RMSNorm as MambaRMSNorm
from .classifier import Classifier

# One-hot channel permutation for reverse complement (A=0,C=1,G=2,T=3 → T=3,G=2,C=1,A=0)
_RC_PERM = [3, 2, 1, 0]


def _reverse_complement_onehot(x: torch.Tensor) -> torch.Tensor:
    """Compute the reverse complement of a one-hot encoded sequence.

    Args:
        x: [B, 4, L]  one-hot (channels: A=0, C=1, G=2, T=3)
    Returns:
        [B, 4, L]  reverse complement
    """
    return x[:, _RC_PERM, :].flip(dims=[2])


class CircStem(nn.Module):
    """
    Two-stage model that mimics the biological back-splicing mechanism.

    Biological motivation
    ─────────────────────
    Back-splicing requires two steps:
      1. Flanking introns find each other via reverse-complementary sequences
         (RCM, e.g. inverted Alu repeats) → they "zip up" into a stable stem,
         physically looping the exon.
      2. The spliceosome recognises the looped splice sites and performs
         back-splicing.

    Model mapping
    ─────────────
    Stage 1 — Stem Scoring (intron pairing)
      upper_intron cross-attends to lower_intron_rc.
      lower_intron_rc is computed inside the model via reverse complement on
      the one-hot tensor — no external data-loader changes needed.
      The resulting L×L attention map is processed by a 2D CNN that learns
      which position pairs are diagnostically important for stem formation.
      → stem_feat: compact summary of pairing quality.

    Stage 2 — Junction Scoring (splice-site recognition)
      upper_exon and lower_exon features (1D CNN) are element-wise gated by a
      projection of stem_feat: a weak stem suppresses junction recognition,
      a strong stem amplifies it.
      → junc_gated: splice-site features conditioned on stem quality.

    Input format  (double-input, one-hot, same as circcnn/circdc)
    ─────────────
      upper_seq: [B, 4, 2L]  upper_intron ([:L]) + upper_exon ([L:])
      lower_seq: [B, 4, 2L]  lower_exon ([:L]) + lower_intron ([L:])
    where L = junction_bps.

    Internal split:
      upper_intron    = upper_seq[:, :, :L]
      upper_exon      = upper_seq[:, :, L:]
      lower_exon      = lower_seq[:, :, :L]
      lower_intron_rc = RC(lower_seq[:, :, L:])   ← computed internally

    Visualisation  (return_aux=True)
    ─────────────────────────────────
      stem_attn_map  [B, L, L]  per-position intron pairing heatmap
      gate           [B, d_junc] stem→junction gate values (0=blocked, 1=open)
      stem_feat      [B, d_stem] stem summary (UMAP / clustering)
      junc_feat      [B, d_junc] junction features before gating
      u_intron_emb   [B, L, d]  upper intron per-position embeddings
      l_intron_emb   [B, L, d]  lower intron RC per-position embeddings
    """

    def __init__(
        self,
        junction_bps: int = 100,
        d_model: int = 128,
        n_heads: int = 4,
        n_mamba_layers: int = 2,
        dropout: float = 0.1,
        # 1D CNN shared intron encoder
        intron_conv_channels: int = 128,
        intron_conv_kernel: int = 7,
        # 2D CNN on stem attention map
        stem_conv_channels: tuple = (16, 32, 64),
        stem_pool_size: int = 2,
        stem_adaptive_size: int = 4,   # AdaptiveAvgPool2d output side
        # 1D CNN junction encoder
        junc_conv_channels: int = 128,
        junc_conv_kernel: int = 7,
        junc_adaptive_size: int = 8,   # AdaptiveAvgPool1d output length
    ):
        super().__init__()
        self.junction_bps = junction_bps

        # ── Stage 1: Intron Encoder — Conv1d(4→d) → Mamba (shared) ─────
        # Input channels = 4 (one-hot ACGT), no pooling (preserves length L)
        self.intron_cnn = nn.Sequential(
            nn.Conv1d(4, intron_conv_channels,
                      kernel_size=intron_conv_kernel,
                      padding=intron_conv_kernel // 2),
            nn.BatchNorm1d(intron_conv_channels),
            nn.ReLU(inplace=True),
        )
        self.intron_proj = nn.Linear(intron_conv_channels, d_model) \
            if intron_conv_channels != d_model else nn.Identity()

        chunk_size = max(1, junction_bps // 10)
        mamba_cfg = Mamba2Config(
            d_model=d_model,
            n_layer=n_mamba_layers,
            d_state=64,
            d_conv=4,
            expand=2,
            headdim=d_model // n_heads,
            chunk_size=chunk_size,
        )
        # Shared Mamba layers: applied to both upper_intron and lower_intron_rc
        self.intron_mamba = nn.ModuleList([
            nn.ModuleDict({
                "mixer": Mamba2(mamba_cfg),
                "norm":  MambaRMSNorm(d_model),
            })
            for _ in range(n_mamba_layers)
        ])

        # Cross-attention: upper_intron queries lower_intron_rc
        self.stem_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )

        # 2D CNN on stem attention map [B, 1, L, L]
        c1, c2, c3 = stem_conv_channels
        self.stem_map_cnn = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stem_pool_size),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stem_pool_size),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # AdaptiveAvgPool makes d_stem independent of junction_bps
        self.stem_adaptive_pool = nn.AdaptiveAvgPool2d(stem_adaptive_size)
        self.d_stem = c3 * stem_adaptive_size * stem_adaptive_size

        # ── Stage 2: Junction Encoder — shared 1D CNN ─────────────────
        # Input channels = 4 (one-hot ACGT)
        self.junction_cnn = nn.Sequential(
            nn.Conv1d(4, junc_conv_channels,
                      kernel_size=junc_conv_kernel,
                      padding=junc_conv_kernel // 2),
            nn.BatchNorm1d(junc_conv_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(junc_conv_channels, junc_conv_channels,
                      kernel_size=junc_conv_kernel,
                      padding=junc_conv_kernel // 2),
            nn.BatchNorm1d(junc_conv_channels),
            nn.ReLU(inplace=True),
        )
        self.junc_adaptive_pool = nn.AdaptiveAvgPool1d(junc_adaptive_size)
        d_junc_single = junc_conv_channels * junc_adaptive_size
        self.d_junc = d_junc_single * 2   # upper_exon + lower_exon

        # ── Gate: stem quality modulates junction recognition ───────────
        # Biologically: strong stem → spliceosome can access junction
        self.gate_proj = nn.Sequential(
            nn.Linear(self.d_stem, self.d_junc),
            nn.Sigmoid(),
        )

        # ── Classifier ──────────────────────────────────────────────────
        self.classifier = Classifier(d_in=self.d_stem + self.d_junc)

        self.dropout = nn.Dropout(dropout)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _encode_intron(self, onehot: torch.Tensor) -> torch.Tensor:
        """Shared CNN → Mamba encoder for an intron sequence.

        Args:
            onehot: [B, 4, L]
        Returns:
            x: [B, L, d_model]  (length-preserving)
        """
        x = self.intron_cnn(onehot)                  # [B, C, L]
        x = self.intron_proj(x.transpose(1, 2))      # [B, L, d_model]
        for layer in self.intron_mamba:
            z, _ = layer["mixer"](layer["norm"](x))
            x = x + z                                 # residual
        return x                                      # [B, L, d_model]

    def _encode_junction(self, onehot: torch.Tensor) -> torch.Tensor:
        """Shared 1D CNN encoder for exonic (junction) sequences.

        Args:
            onehot: [B, 4, L]
        Returns:
            feat: [B, d_junc_single]
        """
        x = self.junction_cnn(onehot)                # [B, C, L']
        x = self.junc_adaptive_pool(x)               # [B, C, junc_adaptive_size]
        return x.view(x.size(0), -1)                 # [B, d_junc_single]

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
            upper_seq: [B, 4, 2L]  one-hot  (upper_intron | upper_exon)
            lower_seq: [B, 4, 2L]  one-hot  (lower_exon  | lower_intron)
            return_aux: if True, also returns a dict of intermediate tensors

        Returns:
            logits [B, 2], and optionally aux dict.
        """
        L = self.junction_bps

        # ── Split ───────────────────────────────────────────────────────
        upper_intron = upper_seq[:, :, :L]      # [B, 4, L] upstream intronic
        upper_exon   = upper_seq[:, :, L:]      # [B, 4, L] upstream exonic (junction)
        lower_exon   = lower_seq[:, :, :L]      # [B, 4, L] downstream exonic (junction)
        lower_intron = lower_seq[:, :, L:]      # [B, 4, L] downstream intronic

        # RC of lower_intron computed inside the model — no external preprocessing
        lower_intron_rc = _reverse_complement_onehot(lower_intron)  # [B, 4, L]

        # ── Stage 1: Stem Scoring ────────────────────────────────────────
        u_i = self._encode_intron(upper_intron)     # [B, L, d_model]
        l_i = self._encode_intron(lower_intron_rc)  # [B, L, d_model]  (shared encoder)

        # Cross-attention: upper_intron queries lower_intron_rc
        # stem_attn_map[b, i, j] = how strongly position i of upper_intron
        #                           pairs with position j of lower_intron_rc
        _, stem_attn_map = self.stem_attn(
            query=u_i, key=l_i, value=l_i,
            need_weights=True,
            average_attn_weights=True,   # [B, L, L] averaged over heads
        )

        # 2D CNN extracts pairing patterns from the stem map
        x_stem = self.stem_map_cnn(stem_attn_map.unsqueeze(1))  # [B, C, H, W]
        x_stem = self.stem_adaptive_pool(x_stem)                 # [B, C, s, s]
        stem_feat = self.dropout(x_stem.view(x_stem.size(0), -1))  # [B, d_stem]

        # ── Stage 2: Junction Scoring ────────────────────────────────────
        ue_feat = self._encode_junction(upper_exon)   # [B, d_junc/2]
        le_feat = self._encode_junction(lower_exon)   # [B, d_junc/2]
        junc_feat = self.dropout(
            torch.cat([ue_feat, le_feat], dim=1)       # [B, d_junc]
        )

        # Gate: how strongly does this stem structure enable junction back-splicing?
        gate = self.gate_proj(stem_feat)               # [B, d_junc], values in (0, 1)
        junc_gated = junc_feat * gate                  # [B, d_junc]

        # ── Fusion & Classification ──────────────────────────────────────
        combined = torch.cat([stem_feat, junc_gated], dim=1)
        logits = self.classifier(combined)

        if not return_aux:
            return logits

        return logits, {
            # ── Primary visualisation targets ──
            "stem_attn_map": stem_attn_map.detach(),  # [B, L, L]  intron pairing heatmap
            "gate":          gate.detach(),            # [B, d_junc] junction gate values

            # ── Embedding spaces (UMAP, clustering) ──
            "stem_feat":     stem_feat.detach(),       # [B, d_stem]
            "junc_feat":     junc_feat.detach(),       # [B, d_junc]

            # ── Per-position intron representations ──
            "u_intron_emb":  u_i.detach(),             # [B, L, d_model]
            "l_intron_emb":  l_i.detach(),             # [B, L, d_model]
        }

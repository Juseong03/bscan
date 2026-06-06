import torch
import torch.nn as nn

from .bscan_seq_lite import BSCANSeqLite
from .classifier import Classifier


class BSCANSeqRCAttn(BSCANSeqLite):
    """
    BSCAN with RC-paired cross-attention summaries.

    The predictive CNN branch remains the same as BSCANSeqLite. RC sequences
    are used only in two biologically matched attention pairs:
      - upper attends to lower_rc
      - lower attends to upper_rc
    """

    def __init__(
        self,
        *args,
        d_model: int = 128,
        dropout_prob: float = 0.2,
        rc_attention_heads: int = 4,
        **kwargs,
    ):
        super().__init__(
            *args,
            d_model=d_model,
            dropout_prob=dropout_prob,
            **kwargs,
        )
        if d_model % rc_attention_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by rc_attention_heads ({rc_attention_heads}).")

        self.rc_cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=rc_attention_heads,
            batch_first=True,
            dropout=dropout_prob,
        )
        self.rc_attn_norm = nn.LayerNorm(4 * d_model)
        self.rc_attn_project = nn.Sequential(
            nn.Linear(4 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout_prob),
        )

        self.fc_input_dim += 2 * d_model
        self.classifier = Classifier(d_in=self.fc_input_dim, dropout=dropout_prob)

    def _rc_attention_features(
        self,
        upper_seq: torch.Tensor,
        lower_seq: torch.Tensor,
        upper_rc_seq: torch.Tensor,
        lower_rc_seq: torch.Tensor,
    ) -> torch.Tensor:
        upper_emb = self._encode(upper_seq)
        lower_emb = self._encode(lower_seq)
        upper_rc_emb = self._encode(upper_rc_seq)
        lower_rc_emb = self._encode(lower_rc_seq)

        upper_ctx, _ = self.rc_cross_attention(
            upper_emb,
            lower_rc_emb,
            lower_rc_emb,
            need_weights=False,
        )
        lower_ctx, _ = self.rc_cross_attention(
            lower_emb,
            upper_rc_emb,
            upper_rc_emb,
            need_weights=False,
        )

        summary = torch.cat(
            [
                upper_ctx.mean(dim=1),
                lower_ctx.mean(dim=1),
                upper_ctx.amax(dim=1),
                lower_ctx.amax(dim=1),
            ],
            dim=1,
        )
        return self.rc_attn_project(self.rc_attn_norm(summary))

    def extract_features(self, upper_seq: torch.Tensor, lower_seq: torch.Tensor):
        upper_rc_seq = self._reverse_complement_tokens(upper_seq)
        lower_rc_seq = self._reverse_complement_tokens(lower_seq)

        upper_conv = self._cnn_features(upper_seq)
        lower_conv = self._cnn_features(lower_seq)
        stem_feat, bp_map, row_max, col_max, stats = self._stem_profile(upper_seq, lower_rc_seq)
        rc_attn_feat = self._rc_attention_features(upper_seq, lower_seq, upper_rc_seq, lower_rc_seq)

        parts = [upper_conv, lower_conv, rc_attn_feat, stem_feat]
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
            "upper_rc_seq": upper_rc_seq.detach(),
            "lower_rc_seq": lower_rc_seq.detach(),
            "rc_attention": rc_attn_feat.detach(),
        }
        if seq_stats is not None:
            aux["sequence_stats"] = seq_stats.detach()
        return features, aux

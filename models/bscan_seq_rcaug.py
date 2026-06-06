import torch
import torch.nn as nn

from .bscan_seq_lite import BSCANSeqLite


class BSCANSeqRCAug(BSCANSeqLite):
    """
    RC-augmented BSCAN.

    The same encoder is applied to upper/lower and their reverse complements.
    For each side, original and RC features are fused by concat/difference/product
    and projected back to one compact side feature.
    """

    def __init__(
        self,
        *args,
        dropout_prob: float = 0.2,
        conv_out_channels2: int = 256,
        **kwargs,
    ):
        super().__init__(
            *args,
            conv_out_channels2=conv_out_channels2,
            dropout_prob=dropout_prob,
            **kwargs,
        )
        self.rc_pair_project = nn.Sequential(
            nn.Conv1d(4 * conv_out_channels2, conv_out_channels2, kernel_size=1),
            nn.BatchNorm1d(conv_out_channels2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
        )

    def _cnn_feature_map(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self._encode(tokens)
        x = self.conv_layer1(embedded.transpose(1, 2))
        return self.conv_layer2(x)

    def _fuse_rc_pair(self, seq_map: torch.Tensor, rc_map: torch.Tensor) -> torch.Tensor:
        fused = torch.cat(
            [
                seq_map,
                rc_map,
                torch.abs(seq_map - rc_map),
                seq_map * rc_map,
            ],
            dim=1,
        )
        return self.rc_pair_project(fused).flatten(start_dim=1)

    def extract_features(self, upper_seq: torch.Tensor, lower_seq: torch.Tensor):
        upper_rc_seq = self._reverse_complement_tokens(upper_seq)
        lower_rc_seq = self._reverse_complement_tokens(lower_seq)

        upper_conv = self._cnn_feature_map(upper_seq)
        lower_conv = self._cnn_feature_map(lower_seq)
        upper_rc_conv = self._cnn_feature_map(upper_rc_seq)
        lower_rc_conv = self._cnn_feature_map(lower_rc_seq)

        upper_feat = self._fuse_rc_pair(upper_conv, upper_rc_conv)
        lower_feat = self._fuse_rc_pair(lower_conv, lower_rc_conv)
        stem_feat, bp_map, row_max, col_max, stats = self._stem_profile(upper_seq, lower_rc_seq)

        parts = [upper_feat, lower_feat]
        cross_attn_feat = None
        if self.use_cross_attention:
            upper_emb = self._encode(upper_seq)
            lower_rc_emb = self._encode(lower_rc_seq)
            upper_ctx, _ = self.cross_attention(
                upper_emb,
                lower_rc_emb,
                lower_rc_emb,
                need_weights=False,
            )
            lower_ctx, _ = self.cross_attention(
                lower_rc_emb,
                upper_emb,
                upper_emb,
                need_weights=False,
            )
            cross_attn_feat = self.cross_attention_norm(
                torch.cat([upper_ctx.mean(dim=1), lower_ctx.mean(dim=1)], dim=1)
            )
            parts.append(cross_attn_feat)

        parts.append(stem_feat)
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
        }
        if cross_attn_feat is not None:
            aux["cross_attention"] = cross_attn_feat.detach()
        if seq_stats is not None:
            aux["sequence_stats"] = seq_stats.detach()
        return features, aux

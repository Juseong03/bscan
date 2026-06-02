import torch
import torch.nn as nn

from .classifier import Classifier
from .mamba2 import Mamba2, Mamba2Config, RMSNorm


class _CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, key, value):
        attn_out, _ = self.attn(query, key, value, need_weights=False)
        x = self.norm1(query + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class BSCANMambaXAttn(nn.Module):
    """
    BSCAN-Mamba-XAttn:
    - shared token embedding
    - per-sequence Mamba encoder
    - explicit upper <-> lower_rc cross-attention
    - explicit intronic stem profile
    """

    def __init__(
        self,
        junction_bps: int = 100,
        d_model: int = 128,
        mamba_layers: int = 2,
        d_head: int = 4,
        d_state: int = 64,
        expand_factor: int = 2,
        d_conv: int = 3,
        n_groups: int = 1,
        activation: str = "silu",
        bias: bool = False,
        conv_bias: bool = True,
        attn_layers: int = 1,
        attn_heads: int = 4,
        attn_ff_dim: int = 256,
        dropout: float = 0.2,
        use_stem_profile: bool = True,
        use_stem_stats: bool = True,
        use_seq_stats: bool = False,
        use_bidirectional_attention: bool = False,
    ):
        super().__init__()
        self.junction_bps = junction_bps
        self.length_seq = 2 * junction_bps
        self.use_stem_profile = use_stem_profile
        self.use_stem_stats = use_stem_stats
        self.use_seq_stats = use_seq_stats
        self.use_bidirectional_attention = use_bidirectional_attention
        self.d_model = d_model

        self.embedding = nn.Embedding(26, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.mamba_config = Mamba2Config(
            d_model=d_model,
            n_layer=mamba_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand_factor,
            headdim=d_head,
            chunk_size=max(1, self.length_seq),
        )
        self.mamba_layers = nn.ModuleList([
            nn.ModuleDict({
                "mixer": Mamba2(self.mamba_config),
                "norm": RMSNorm(d_model),
            })
            for _ in range(mamba_layers)
        ])

        if d_model % attn_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by attn_heads ({attn_heads}).")
        self.cross_attn_layers = nn.ModuleList([
            _CrossAttnBlock(d_model, attn_heads, attn_ff_dim, dropout)
            for _ in range(attn_layers)
        ])

        self.base_token_ids = {6: 0, 7: 1, 8: 2, 9: 3}
        rc_lookup = torch.arange(26, dtype=torch.long)
        rc_lookup[6] = 9
        rc_lookup[9] = 6
        rc_lookup[7] = 8
        rc_lookup[8] = 7
        self.register_buffer("rc_lookup", rc_lookup, persistent=False)

        profile_dim = 2 * junction_bps if use_stem_profile else 0
        stats_dim = 4 if use_stem_stats else 0
        seq_stats_dim = 20 if use_seq_stats else 0
        attn_dim = d_model * (2 if use_bidirectional_attention else 1)
        self.fc_input_dim = d_model * 2 + attn_dim + profile_dim + stats_dim + seq_stats_dim
        self.classifier = Classifier(d_in=self.fc_input_dim, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def _reverse_complement_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.rc_lookup[tokens.flip(dims=[1])]

    def _token_to_one_hot(self, tokens: torch.Tensor) -> torch.Tensor:
        x = torch.zeros(tokens.size(0), 4, tokens.size(1), device=tokens.device)
        for token_id, channel in self.base_token_ids.items():
            x[:, channel, :] = (tokens == token_id).to(x.dtype)
        return x

    def _encode(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for layer in self.mamba_layers:
            z, _ = layer["mixer"](layer["norm"](x))
            x = x + z
        return x

    def _stem_profile(self, upper_seq: torch.Tensor, lower_rc_seq: torch.Tensor):
        L = self.junction_bps
        upper_intron = upper_seq[:, :L]
        lower_intron_rc = lower_rc_seq[:, :L]

        u_oh = self._token_to_one_hot(upper_intron)
        l_rc_oh = self._token_to_one_hot(lower_intron_rc)
        bp_map = torch.bmm(u_oh.transpose(1, 2), l_rc_oh)

        row_max = bp_map.max(dim=-1).values
        col_max = bp_map.max(dim=-2).values
        stats = torch.stack(
            [
                bp_map.mean(dim=(1, 2)),
                bp_map.amax(dim=(1, 2)),
                row_max.mean(dim=1),
                col_max.mean(dim=1),
            ],
            dim=1,
        )

        parts = []
        if self.use_stem_profile:
            parts.extend([row_max, col_max])
        if self.use_stem_stats:
            parts.append(stats)
        stem_feat = torch.cat(parts, dim=1) if parts else bp_map.new_zeros(bp_map.size(0), 0)
        return self.drop(stem_feat), bp_map, row_max, col_max, stats

    def _sequence_stats(self, upper_seq: torch.Tensor, lower_seq: torch.Tensor) -> torch.Tensor:
        L = self.junction_bps
        regions = [
            upper_seq[:, :L],
            upper_seq[:, L:],
            lower_seq[:, :L],
            lower_seq[:, L:],
        ]

        parts = []
        for region in regions:
            base_freq = self._token_to_one_hot(region).mean(dim=2)
            gc_frac = base_freq[:, 1:3].sum(dim=1, keepdim=True)
            parts.extend([base_freq, gc_frac])
        return torch.cat(parts, dim=1)

    def extract_features(self, upper_seq: torch.Tensor, lower_seq: torch.Tensor):
        lower_rc_seq = self._reverse_complement_tokens(lower_seq)

        upper_enc = self._encode(upper_seq)
        lower_enc = self._encode(lower_seq)
        upper_rc_enc = self._encode(self._reverse_complement_tokens(upper_seq))
        lower_rc_enc = self._encode(lower_rc_seq)

        upper_pool = upper_enc.mean(dim=1)
        lower_pool = lower_enc.mean(dim=1)

        attn_feat = None
        upper_ctx = upper_enc
        lower_ctx = lower_rc_enc
        for block in self.cross_attn_layers:
            upper_ctx = block(upper_ctx, lower_ctx, lower_ctx)
            if self.use_bidirectional_attention:
                lower_ctx = block(lower_ctx, upper_rc_enc, upper_rc_enc)

        attn_parts = [upper_ctx.mean(dim=1)]
        if self.use_bidirectional_attention:
            attn_parts.append(lower_ctx.mean(dim=1))
        attn_feat = torch.cat(attn_parts, dim=1)

        stem_feat, bp_map, row_max, col_max, stats = self._stem_profile(upper_seq, lower_rc_seq)

        parts = [upper_pool, lower_pool, attn_feat, stem_feat]
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
            "upper_rc_seq": upper_rc_enc.detach(),
            "upper_pool": upper_pool.detach(),
            "lower_pool": lower_pool.detach(),
            "attn_feat": attn_feat.detach(),
        }
        if seq_stats is not None:
            aux["sequence_stats"] = seq_stats.detach()
        return features, aux

    def forward(self, upper_seq, lower_seq, return_aux: bool = False):
        features, aux = self.extract_features(upper_seq, lower_seq)
        logits = self.classifier(features)
        if not return_aux:
            return logits
        return logits, aux

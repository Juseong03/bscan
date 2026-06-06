import torch
import torch.nn as nn

from .bscan_seq_lite import BSCANSeqLite
from .classifier import Classifier
from .mamba2 import Mamba2, Mamba2Config, RMSNorm


class BSCANSeqMambaAux(BSCANSeqLite):
    """
    BSCAN with a lightweight auxiliary Mamba branch.

    The original BSCAN CNN + explicit stem profile stays unchanged. Mamba is
    used only to add a compact sequential summary, limiting the risk of
    weakening local/directional BS-LS signals.
    """

    def __init__(
        self,
        *args,
        mamba_d_model: int = 64,
        mamba_layers: int = 1,
        mamba_d_state: int = 32,
        mamba_d_head: int = 4,
        mamba_expand: int = 2,
        mamba_d_conv: int = 3,
        mamba_summary_dim: int = 128,
        dropout_prob: float = 0.2,
        **kwargs,
    ):
        super().__init__(
            *args,
            dropout_prob=dropout_prob,
            **kwargs,
        )
        self.mamba_embedding = nn.Embedding(26, mamba_d_model)
        nn.init.normal_(self.mamba_embedding.weight, std=0.02)
        mamba_config = Mamba2Config(
            d_model=mamba_d_model,
            n_layer=mamba_layers,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            headdim=mamba_d_head,
            chunk_size=max(1, self.length_seq),
        )
        self.mamba_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "mixer": Mamba2(mamba_config),
                        "norm": RMSNorm(mamba_d_model),
                    }
                )
                for _ in range(mamba_layers)
            ]
        )
        self.mamba_project = nn.Sequential(
            nn.LayerNorm(4 * mamba_d_model),
            nn.Linear(4 * mamba_d_model, mamba_summary_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
        )
        self.fc_input_dim += mamba_summary_dim
        self.classifier = Classifier(d_in=self.fc_input_dim, dropout=dropout_prob)

    def _mamba_encode(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.mamba_embedding(tokens)
        for layer in self.mamba_layers:
            z, _ = layer["mixer"](layer["norm"](x))
            x = x + z
        return x

    def _mamba_summary(self, upper_seq: torch.Tensor, lower_seq: torch.Tensor) -> torch.Tensor:
        upper = self._mamba_encode(upper_seq)
        lower = self._mamba_encode(lower_seq)
        summary = torch.cat(
            [
                upper.mean(dim=1),
                lower.mean(dim=1),
                upper.amax(dim=1),
                lower.amax(dim=1),
            ],
            dim=1,
        )
        return self.mamba_project(summary)

    def extract_features(self, upper_seq: torch.Tensor, lower_seq: torch.Tensor):
        features, aux = super().extract_features(upper_seq, lower_seq)
        mamba_feat = self._mamba_summary(upper_seq, lower_seq)
        aux["mamba_aux"] = mamba_feat.detach()
        return torch.cat([features, mamba_feat], dim=1), aux

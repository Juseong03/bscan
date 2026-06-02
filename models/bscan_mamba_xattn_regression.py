from .bscan_mamba_xattn import BSCANMambaXAttn
import torch.nn as nn


class RegressionHead(nn.Module):
    def __init__(self, d_in: int, hidden_dims=(512, 256), dropout: float = 0.2):
        super().__init__()
        layers = [nn.LayerNorm(d_in)]
        prev_dim = d_in
        for idx, hidden_dim in enumerate(hidden_dims):
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout if idx == 0 else dropout * 0.5),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BSCANMambaXAttnRegression(BSCANMambaXAttn):
    def __init__(
        self,
        *args,
        d_hidden: int = 512,
        d_hidden2: int = 256,
        reg_dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__(*args, dropout=reg_dropout, **kwargs)
        self.classifier = RegressionHead(
            d_in=self.fc_input_dim,
            hidden_dims=(d_hidden, d_hidden2),
            dropout=reg_dropout,
        )

    def forward(self, upper_seq, lower_seq, return_aux: bool = False):
        out = super().forward(upper_seq, lower_seq, return_aux=return_aux)
        if not return_aux:
            return out.squeeze(-1)
        logits, aux = out
        return logits.squeeze(-1), aux

from .bscan_v2 import BSCANv2
from .classifier import Classifier


class BSCANv2Regression(BSCANv2):
    """Regression head for BSCAN-v2 expression prediction."""

    def __init__(self, *args, d_hidden: int = 128, reg_dropout: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = Classifier(
            d_in=self.fc_input_dim,
            d_hiddens=[d_hidden],
            dropout=reg_dropout,
            n_classes=1,
        )

    def forward(self, upper_seq, lower_seq, lower_rc_seq, return_aux: bool = False):
        out = super().forward(upper_seq, lower_seq, lower_rc_seq, return_aux=return_aux)
        if not return_aux:
            return out.squeeze(-1)
        logits, aux = out
        return logits.squeeze(-1), aux

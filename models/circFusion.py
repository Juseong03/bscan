import torch
import torch.nn as nn

from .circATTRCM import CircATTRCM
from .circMamba import circMamba


class CircFusionMambaATTRCM(nn.Module):
    """
    Learnable fusion of two "ours" models:
    - CircATTRCM: interaction map (2D conv) style
    - circMamba: sequence mixing + CNN + cross-attention style

    Both consume the same inputs from `DataSetPrep.tensor_for_pretrained(..., rc=True)`:
      upper_seq, lower_seq, lower_rc_seq  (token ids)

    We fuse the two models at the *logit* level with a learnable scalar gate:
      fused = sigmoid(alpha) * logits_attrcm + (1 - sigmoid(alpha)) * logits_mamba

    This is a simple but strong baseline; if it helps, we can later fuse intermediate features.
    """

    def __init__(
        self,
        # submodel kwargs (kept minimal; can be expanded as needed)
        attrcm_kwargs: dict | None = None,
        mamba_kwargs: dict | None = None,
        init_alpha: float = 0.0,  # sigmoid(0)=0.5 -> equal weighting initially
    ):
        super().__init__()
        self.attrcm = CircATTRCM(**(attrcm_kwargs or {}))
        self.mamba = circMamba(**(mamba_kwargs or {}))

        # Learnable fusion weight
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, upper_seq, lower_seq, lower_rc_seq):
        logits_attrcm = self.attrcm(upper_seq, lower_seq, lower_rc_seq)
        logits_mamba = self.mamba(upper_seq, lower_seq, lower_rc_seq)

        # Some models may return (logits, aux); keep this robust.
        if isinstance(logits_attrcm, (tuple, list)):
            logits_attrcm = logits_attrcm[0]
        if isinstance(logits_mamba, (tuple, list)):
            logits_mamba = logits_mamba[0]

        w = torch.sigmoid(self.alpha)
        return w * logits_attrcm + (1.0 - w) * logits_mamba



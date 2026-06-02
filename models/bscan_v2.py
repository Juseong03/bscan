import torch
import torch.nn as nn
import torch.nn.functional as F
from multimolecule import RnaBertModel, RnaErnieModel

from .classifier import Classifier
from .circCNNATT import MultiLayerAttentionBlock


class BSCANv2(nn.Module):
    """
    BSCAN-v2: CircCNNATT backbone with an explicit intronic stem branch.

    Inputs are the same token tensors used by CircCNNATT:
      upper_seq    [B, 2L] = upper intron + upper exon
      lower_seq    [B, 2L] = lower exon + lower intron
      lower_rc_seq [B, 2L] = reverse-complement(lower_seq)

    The predictive backbone is kept close to CircCNNATT. The added biological
    branch computes a Watson-Crick pairing map between upper intron and reverse
    complemented lower intron, then exposes both local 2D stem features and
    per-position pairing profiles to the classifier.
    """

    def __init__(
        self,
        junction_bps: int = 100,
        d_model: int = 128,
        conv_out_channels1: int = 256,
        conv_kernel_size1: int = 11,
        conv_out_channels2: int = 256,
        conv_kernel_size2: int = 11,
        length_seq: int | None = None,
        n_heads: int = 4,
        ff_dim: int = 128,
        n_attntions: int = 3,
        dropout_prob: float = 0.1,
        pretrained_model: str = "rnaernie",
        use_pretrained: bool = False,
        stem_channels: tuple[int, int, int] = (16, 32, 64),
        stem_adaptive: int = 4,
        use_stem_profile: bool = True,
    ):
        super().__init__()
        self.junction_bps = junction_bps
        self.length_seq = length_seq or (2 * junction_bps)
        self.use_pretrained = use_pretrained
        self.use_stem_profile = use_stem_profile

        if use_pretrained:
            if pretrained_model.lower() == "rnaernie":
                pretrained = RnaErnieModel.from_pretrained("multimolecule/rnaernie")
            elif pretrained_model.lower() == "rnabert":
                pretrained = RnaBertModel.from_pretrained("multimolecule/rnabert")
            else:
                raise ValueError(f"Invalid pretrained model: {pretrained_model}")
            self.embeddings = pretrained.embeddings
            self.encoder = None
            for param in self.embeddings.parameters():
                param.requires_grad = False
            self.to_proj = nn.Sequential(nn.Linear(768, d_model), nn.GELU())
        else:
            self.embeddings = nn.Embedding(26, d_model)
            nn.init.normal_(self.embeddings.weight, std=0.02)
            self.to_proj = nn.Identity()

        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(
                in_channels=d_model,
                out_channels=conv_out_channels1,
                kernel_size=conv_kernel_size1,
                padding=(conv_kernel_size1 - 1) // 2,
            ),
            nn.BatchNorm1d(conv_out_channels1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(
                in_channels=conv_out_channels1,
                out_channels=conv_out_channels2,
                kernel_size=conv_kernel_size2,
                padding=(conv_kernel_size2 - 1) // 2,
            ),
            nn.BatchNorm1d(conv_out_channels2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )
        self.out_dim = self.length_seq // (4 * 4)

        self.attention_block = MultiLayerAttentionBlock(
            embed_dim=d_model,
            num_heads=n_heads,
            ff_dim=ff_dim,
            num_layers=n_attntions,
            dropout_prob=dropout_prob,
        )

        c1, c2, c3 = stem_channels
        self.stem_cnn = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(stem_adaptive),
        )
        self.drop = nn.Dropout(dropout_prob)

        stem_dim = c3 * stem_adaptive * stem_adaptive
        profile_dim = 2 * junction_bps if use_stem_profile else 0
        self.fc_input_dim = (
            (conv_out_channels2 * self.out_dim * 2)
            + d_model
            + stem_dim
            + profile_dim
        )
        self.classifier = Classifier(d_in=self.fc_input_dim)

        # RNAErnie/RNABert token ids after DataSetPrep.tensor_for_pretrained.
        # DNA T is tokenized as RNA U.
        self.base_token_ids = {6: 0, 7: 1, 8: 2, 9: 3}

    def _token_to_one_hot(self, tokens: torch.Tensor) -> torch.Tensor:
        x = torch.zeros(tokens.size(0), 4, tokens.size(1), device=tokens.device)
        for token_id, channel in self.base_token_ids.items():
            x[:, channel, :] = (tokens == token_id).to(x.dtype)
        return x

    def _stem_features(self, upper_seq: torch.Tensor, lower_rc_seq: torch.Tensor):
        L = self.junction_bps
        upper_intron = upper_seq[:, :L]
        lower_intron_rc = lower_rc_seq[:, :L]

        u_oh = self._token_to_one_hot(upper_intron)
        l_rc_oh = self._token_to_one_hot(lower_intron_rc)
        bp_map = torch.bmm(u_oh.transpose(1, 2), l_rc_oh)

        stem_map_feat = self.stem_cnn(bp_map.unsqueeze(1)).flatten(start_dim=1)
        parts = [self.drop(stem_map_feat)]
        row_max = bp_map.max(dim=-1).values
        col_max = bp_map.max(dim=-2).values
        if self.use_stem_profile:
            parts.extend([row_max, col_max])
        return torch.cat(parts, dim=1), bp_map, row_max, col_max

    def forward(self, upper_seq, lower_seq, lower_rc_seq, return_aux: bool = False):
        upper_embedded = self.to_proj(self.embeddings(upper_seq))
        lower_embedded = self.to_proj(self.embeddings(lower_seq))
        lower_reverse_embedded = self.to_proj(self.embeddings(lower_rc_seq))

        upper_conv = self.conv_layer1(upper_embedded.transpose(1, 2))
        upper_conv = self.conv_layer2(upper_conv).flatten(start_dim=1)

        lower_conv = self.conv_layer1(lower_embedded.transpose(1, 2))
        lower_conv = self.conv_layer2(lower_conv).flatten(start_dim=1)

        attention_output = self.attention_block(
            upper_embedded,
            lower_reverse_embedded,
            lower_embedded,
        )
        stem_feat, bp_map, row_max, col_max = self._stem_features(upper_seq, lower_rc_seq)

        features = torch.cat((upper_conv, lower_conv, attention_output[:, 0, :], stem_feat), dim=1)
        logits = self.classifier(features)

        if not return_aux:
            return logits
        return logits, {
            "bp_map": bp_map.detach(),
            "row_max": row_max.detach(),
            "col_max": col_max.detach(),
            "stem_feat": stem_feat.detach(),
            "attention_cls": attention_output[:, 0, :].detach(),
        }

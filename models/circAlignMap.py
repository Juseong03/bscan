import torch
import torch.nn as nn
import torch.nn.functional as F

from multimolecule import RnaErnieModel, RnaBertModel

from .classifier import Classifier


class CircAlignMap(nn.Module):
    """
    Single-block "alignment map" model:
      tokens -> (pretrained embeddings) -> projection -> cross-attention (upper -> lower_rc)
      -> attention weights matrix (alignment map) -> 2D CNN -> classifier

    Motivation:
    - Use the learned cross-attention weights as an explicit interaction map (easy to visualize).
    - Avoid a multi-head / multi-branch design while still learning end-to-end pairing structure.

    Inputs (same as other pretrained models in this repo):
      upper_seq, lower_seq, lower_rc_seq   (token ids, shape [B, L])

    Notes:
    - We use embeddings (and optionally encoder) from a pretrained RNA model, frozen by default.
    - `return_aux=True` returns attention maps and intermediate embeddings for visualization (heatmaps/UMAP).
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        pretrained_model: str = "rnaernie",
        freeze_pretrained: bool = True,
        use_encoder: bool = False,
        use_pretrained: bool = False, # Change default to False
        # conv settings for map
        conv_channels: tuple[int, int, int] = (16, 32, 64),
        conv_kernel: int = 3,
        pool: int = 2,
        length_seq: int = 200,
    ):
        super().__init__()
        self.use_pretrained = use_pretrained

        if use_pretrained:
            if pretrained_model.lower() == "rnaernie":
                pretrained = RnaErnieModel.from_pretrained("multimolecule/rnaernie")
            elif pretrained_model.lower() == "rnabert":
                pretrained = RnaBertModel.from_pretrained("multimolecule/rnabert")
            else:
                raise ValueError(f"Invalid pretrained_model: {pretrained_model}")

            self.embeddings = pretrained.embeddings
            self.use_encoder = use_encoder
            if use_encoder:
                self.encoder = pretrained.encoder
            else:
                self.encoder = None

            if freeze_pretrained:
                for p in self.embeddings.parameters():
                    p.requires_grad = False
                if self.encoder is not None:
                    for p in self.encoder.parameters():
                        p.requires_grad = False
            
            # Pretrained embedding dim is 768 for these models
            self.to_proj = nn.Sequential(nn.Linear(768, d_model), nn.GELU(), nn.Dropout(dropout))
        else:
            self.embeddings = nn.Embedding(26, d_model)
            self.use_encoder = False
            self.encoder = None
            self.to_proj = nn.Identity()

        # Cross-attention. Use batch_first=True for clarity.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        # 2D CNN over alignment map
        c1, c2, c3 = conv_channels
        self.convblock = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=conv_kernel, padding=conv_kernel // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
            nn.Conv2d(c1, c2, kernel_size=conv_kernel, padding=conv_kernel // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
            nn.Conv2d(c2, c3, kernel_size=conv_kernel, padding=conv_kernel // 2),
            nn.ReLU(inplace=True),
        )

        # Infer conv output dim for classifier
        self.fc_input_dim = self._infer_conv_output_dim(length_seq=length_seq)
        self.classifier = Classifier(d_in=self.fc_input_dim)

    def _infer_conv_output_dim(self, length_seq: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, length_seq, length_seq)
            out = self.convblock(dummy)
            return int(out.numel())

    def _encode_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, L)
        x = self.embeddings(token_ids)  # (B, L, 768)
        if self.use_encoder and self.encoder is not None:
            x = self.encoder(x)["last_hidden_state"]
        return self.to_proj(x)  # (B, L, d_model)

    def forward(self, upper_seq, lower_seq, lower_rc_seq, *, return_aux: bool = False):
        # Encode
        upper = self._encode_tokens(upper_seq)
        lower = self._encode_tokens(lower_seq)
        lower_rc = self._encode_tokens(lower_rc_seq)

        # Cross-attention: query=upper, key=lower_rc, value=lower
        # Use default average_attn_weights=True for better compatibility and memory
        _, attn_map = self.cross_attn(
            query=upper,
            key=lower_rc,
            value=lower,
            need_weights=True,
        )
        # attn_map: (B, L, L)

        # 2D conv expects (B, 1, L, L)
        x = self.convblock(attn_map.unsqueeze(1))
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)

        if not return_aux:
            return logits

        # Return things useful for visualization:
        # - attn_map: alignment heatmap
        # - pooled embeddings: UMAP candidates
        aux = {
            "attn_map": attn_map.detach(),
            "upper_emb": upper.mean(dim=1).detach(),
            "lower_emb": lower.mean(dim=1).detach(),
            "lower_rc_emb": lower_rc.mean(dim=1).detach(),
        }
        return logits, aux



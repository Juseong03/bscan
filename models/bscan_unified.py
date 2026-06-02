
import torch
import torch.nn as nn
import torch.nn.functional as F
from multimolecule import RnaErnieModel, RnaBertModel, RnaFmModel, RnaMsmModel
from .classifier import Classifier

_RC_PERM = [3, 2, 1, 0]   # A=0,C=1,G=2,T=3  ->  T,G,C,A


class _CNNAdapter(nn.Module):
    """Lightweight depthwise-separable CNN adapter: 2 layers, residual.

    Captures local sequence motifs (splice-site k-mers) from FM embeddings.
    Input/output: [B, L, d_model] (batch-first).
    """
    def __init__(self, d_model: int, n_layers: int = 2, kernel: int = 7):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Sequential(
                # depthwise
                nn.Conv1d(d_model, d_model, kernel, padding=kernel // 2, groups=d_model),
                # pointwise
                nn.Conv1d(d_model, d_model, 1),
                nn.GELU(),
            ))
        self.layers = nn.ModuleList(layers)
        self.norms  = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        for conv, norm in zip(self.layers, self.norms):
            residual = x
            h = conv(x.transpose(1, 2)).transpose(1, 2)   # [B, L, d]
            x = norm(residual + h)
        return x


class _MambaAdapter(nn.Module):
    """Shallow Mamba SSM adapter: 1 ResidualBlock (Norm→Mamba→Add).

    Captures sequential/positional patterns within each sequence.
    Input/output: [B, L, d_model] (batch-first).
    """
    def __init__(self, d_model: int, n_layers: int = 1, d_state: int = 16, expand: int = 2):
        super().__init__()
        from .mamba import ResidualBlock, ModelArgs
        args = ModelArgs(
            d_model=d_model,
            n_layer=n_layers,
            vocab_size=1,          # unused — adapter doesn't need embedding
            d_state=d_state,
            expand=expand,
        )
        self.blocks = nn.ModuleList([ResidualBlock(args) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

def _rc_onehot(x: torch.Tensor) -> torch.Tensor:
    """Reverse complement for one-hot tensor [B, 4, L]"""
    return x[:, _RC_PERM, :].flip(dims=[2])

def _bp_map(u_oh: torch.Tensor, l_rc_oh: torch.Tensor) -> torch.Tensor:
    """Hard Watson-Crick base-pairing: [B, L, 4] @ [B, 4, L] -> [B, L, L]"""
    return torch.bmm(u_oh.transpose(1, 2), l_rc_oh)

class _CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, key, value, need_weights: bool = False):
        attn_out, attn_w = self.attn(query, key, value, need_weights=need_weights, average_attn_weights=True)
        query = self.norm1(query + attn_out)
        query = self.norm2(query + self.ff(query))
        return query, attn_w

class BSCANUnified(nn.Module):
    def __init__(
        self,
        encoder_type: str = 'onehot',
        d_model: int = 128,
        junction_bps: int = 100,
        use_cached: bool = False,
        # Branches
        use_cnn: bool = True,
        use_stem: bool = True,
        use_attn: bool = True,
        # FM adapter (optional post-projection refinement)
        adapter_type: str | None = None,   # None | 'cnn' | 'mamba'
        adapter_layers: int = 2,           # CNN: 2 layers; Mamba: 1 layer recommended
        # Hyperparams
        n_heads: int = 4,
        n_attn_layers: int = 2,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        self.junction_bps = junction_bps
        self.use_cached = use_cached
        self.use_cnn = use_cnn
        self.use_stem = use_stem
        self.use_attn = use_attn
        self.adapter_type = adapter_type
        L = junction_bps

        # 1. Foundation Model Encoder & Projection
        if self.encoder_type == 'onehot':
            self.embedding = nn.Embedding(26, d_model)
            self.proj = nn.Identity()
        elif not use_cached:
            # Live inference mode (slow)
            if self.encoder_type == 'rnaernie':
                fm = RnaErnieModel.from_pretrained('multimolecule/rnaernie')
            elif self.encoder_type == 'rnabert':
                fm = RnaBertModel.from_pretrained('multimolecule/rnabert')
            elif self.encoder_type == 'rnafm':
                fm = RnaFmModel.from_pretrained('multimolecule/rnafm')
            elif self.encoder_type == 'rnamsm':
                fm = RnaMsmModel.from_pretrained('multimolecule/rnamsm')
            else:
                raise ValueError(f"Unsupported encoder type: {encoder_type}")
            
            self.fm_model = fm
            for param in self.fm_model.parameters(): param.requires_grad = False
            
            hidden_dim = fm.config.hidden_size
            self.proj = nn.Sequential(nn.Linear(hidden_dim, d_model), nn.GELU())
        else:
            # Cached mode: skip FM, just define Projection with correct dim
            fm_dims = {'rnaernie': 768, 'rnabert': 120, 'rnafm': 640, 'rnamsm': 768}
            hidden_dim = fm_dims.get(self.encoder_type, 768)
            self.proj = nn.Sequential(nn.Linear(hidden_dim, d_model), nn.GELU())

        # 1b. Optional FM adapter (applied after projection, before branches)
        self.fm_adapter: nn.Module | None = None
        if adapter_type == 'cnn':
            self.fm_adapter = _CNNAdapter(d_model, n_layers=adapter_layers)
        elif adapter_type == 'mamba':
            _mamba_layers = max(1, adapter_layers // 2)   # keep it truly shallow
            self.fm_adapter = _MambaAdapter(d_model, n_layers=_mamba_layers)
        elif adapter_type is not None:
            raise ValueError(f"adapter_type must be None, 'cnn', or 'mamba', got {adapter_type!r}")

        # 2. Branch A: CNN (1D-CNN on projected features)
        if use_cnn:
            self.cnn = nn.Sequential(
                nn.Conv1d(d_model, 256, 11, padding=5),
                nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                nn.MaxPool1d(4),
                nn.Conv1d(256, 128, 11, padding=5),
                nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(8)
            )

        # 3. Branch B: Stem Map (WC base-pairing)
        if use_stem:
            self.stem_cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4)
            )

        # 4. Branch C: Cross-Attention
        if use_attn:
            self.attn_layers = nn.ModuleList([
                _CrossAttnBlock(d_model, n_heads, dropout)
                for _ in range(n_attn_layers)
            ])

        # 5. Dynamic Classifier Input Calculation
        total_d = 0
        if use_cnn: total_d += 128 * 8 * 2 # Upper + Lower
        if use_stem: total_d += 64 * 4 * 4 + 2 * L # Stem features + row/col max
        if use_attn: total_d += d_model # Global context

        self.classifier = Classifier(d_in=total_d, d_hiddens=[256], dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def _tokens_to_onehot(self, tokens):
        # A=6, C=7, G=8, U/T=9 in rnaernie/multimolecule vocab
        # Let's map to standard 0,1,2,3 for Stem branch
        B, L = tokens.size()
        oh = torch.zeros(B, 4, L, device=tokens.device)
        for i, tid in enumerate([6, 7, 8, 9]):
            oh[:, i, :] = (tokens == tid).float()
        return oh

    def _get_fm_embedding(self, tokens_or_emb):
        # Determine the target dtype (usually float32 from the projection layer)
        target_dtype = next(self.proj.parameters()).dtype if hasattr(self.proj, 'parameters') and list(self.proj.parameters()) else torch.float32
        
        if self.use_cached or self.encoder_type == 'onehot':
            if self.encoder_type == 'onehot':
                # tokens_or_emb are indices
                return self.proj(self.embedding(tokens_or_emb).to(target_dtype))
            
            # tokens_or_emb are cached hidden states (Half or Float)
            # Crop special tokens if length is 202 (CLS at 0, SEP at 201)
            if tokens_or_emb.dim() == 3 and tokens_or_emb.size(1) == 202:
                tokens_or_emb = tokens_or_emb[:, 1:201, :]
            
            return self.proj(tokens_or_emb.to(target_dtype))
        else:
            # Live inference mode
            with torch.no_grad():
                out = self.fm_model(input_ids=tokens_or_emb.long())
                if isinstance(out, tuple): out = out[0]
                elif hasattr(out, 'last_hidden_state'): out = out.last_hidden_state
            return self.proj(out.to(target_dtype))

    def forward(self, upper, lower, lower_rc=None, upper_oh=None, lower_rc_oh=None, return_aux=False):
        # 1. Get Projected Features
        u_feat = self._get_fm_embedding(upper)  # [B, 2L, d_model]
        l_feat = self._get_fm_embedding(lower)

        # 1b. FM adapter (optional local/sequential refinement)
        if self.fm_adapter is not None:
            u_feat = self.fm_adapter(u_feat)
            l_feat = self.fm_adapter(l_feat)
        
        # 2. Prep for Stem (Branch B)
        # If one-hot tensors not provided (e.g. one-hot mode or live inference), 
        # extract them from tokens.
        L = self.junction_bps
        if upper_oh is None:
            # We need intron only for Stem branch: upper_tokens[:, :L]
            upper_oh = self._tokens_to_onehot(upper[:, :L])
        if lower_rc_oh is None:
            if lower_rc is not None:
                # lower_rc tokens provided: lower_rc_tokens[:, L:] is lower_rc intron
                lower_rc_oh = self._tokens_to_onehot(lower_rc[:, L:])
            else:
                # Fallback: compute from lower tokens
                l_oh_int = self._tokens_to_onehot(lower[:, L:])
                lower_rc_oh = _rc_onehot(l_oh_int)

        parts = []
        aux = {}

        # Branch A: CNN
        if self.use_cnn:
            cnn_u = self.cnn(u_feat.transpose(1, 2)).view(u_feat.size(0), -1)
            cnn_l = self.cnn(l_feat.transpose(1, 2)).view(l_feat.size(0), -1)
            parts.extend([cnn_u, cnn_l])

        # Branch B: Stem
        if self.use_stem:
            # We assume upper_oh and lower_rc_oh are provided if in cached mode
            # upper_oh: [B, 4, L] (upper intron)
            # lower_rc_oh: [B, 4, L] (lower rc intron)
            bp = _bp_map(upper_oh, lower_rc_oh)
            row_max = bp.max(dim=-1).values
            col_max = bp.max(dim=-2).values
            x_stem = self.stem_cnn(bp.unsqueeze(1)).view(upper.size(0), -1)
            parts.extend([x_stem, row_max, col_max])
            if return_aux: aux['bp_map'] = bp

        # Branch C: Attention
        if self.use_attn:
            attn_feat = u_feat
            attn_weights_list = []
            for layer in self.attn_layers:
                attn_feat, attn_w = layer(attn_feat, l_feat, l_feat,
                                          need_weights=return_aux)
                if return_aux and attn_w is not None:
                    attn_weights_list.append(attn_w.detach().cpu())
            parts.append(attn_feat.mean(dim=1))
            if return_aux:
                aux['attn_weights'] = attn_weights_list  # list of [B, L, L] per layer

        # Final Fusion
        combined = torch.cat(parts, dim=1)
        logits = self.classifier(self.drop(combined))

        if return_aux:
            aux['combined'] = combined.detach().cpu()
            return logits, aux
        return logits


class BSCANUnifiedEmbedOnly(nn.Module):
    """BSCAN using only frozen RNA foundation-model token embeddings.

    This skips the contextual encoder and cached hidden-state files. It tests
    whether the FM token embedding table itself adds signal beyond one-hot.
    """

    def __init__(
        self,
        encoder_type: str,
        d_model: int = 128,
        junction_bps: int = 100,
        use_cnn: bool = True,
        use_stem: bool = True,
        use_attn: bool = True,
        n_heads: int = 4,
        n_attn_layers: int = 2,
        dropout: float = 0.3,
        freeze_embedding: bool = True,
        **kwargs
    ):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        self.junction_bps = junction_bps
        self.use_cnn = use_cnn
        self.use_stem = use_stem
        self.use_attn = use_attn
        L = junction_bps

        word_weight = self._load_word_embedding(self.encoder_type)
        self.embedding = nn.Embedding.from_pretrained(word_weight, freeze=freeze_embedding)
        hidden_dim = word_weight.size(1)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, d_model), nn.GELU())

        if use_cnn:
            self.cnn = nn.Sequential(
                nn.Conv1d(d_model, 256, 11, padding=5),
                nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                nn.MaxPool1d(4),
                nn.Conv1d(256, 128, 11, padding=5),
                nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(8)
            )

        if use_stem:
            self.stem_cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4)
            )

        if use_attn:
            self.attn_layers = nn.ModuleList([
                _CrossAttnBlock(d_model, n_heads, dropout)
                for _ in range(n_attn_layers)
            ])

        total_d = 0
        if use_cnn: total_d += 128 * 8 * 2
        if use_stem: total_d += 64 * 4 * 4 + 2 * L
        if use_attn: total_d += d_model

        self.classifier = Classifier(d_in=total_d, d_hiddens=[256], dropout=dropout)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _load_word_embedding(encoder_type: str) -> torch.Tensor:
        if encoder_type.startswith('random'):
            random_dims = {'random120': 120, 'random640': 640, 'random768': 768}
            hidden_dim = random_dims.get(encoder_type)
            if hidden_dim is None:
                raise ValueError(f"Unsupported random embedding type: {encoder_type}")
            return torch.randn(26, hidden_dim) * 0.02
        if encoder_type == 'rnaernie':
            model = RnaErnieModel.from_pretrained('multimolecule/rnaernie')
        elif encoder_type == 'rnabert':
            model = RnaBertModel.from_pretrained('multimolecule/rnabert')
        elif encoder_type == 'rnafm':
            model = RnaFmModel.from_pretrained('multimolecule/rnafm')
        elif encoder_type == 'rnamsm':
            model = RnaMsmModel.from_pretrained('multimolecule/rnamsm')
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

        embeddings = getattr(model, 'embeddings')
        word_embeddings = getattr(embeddings, 'word_embeddings')
        weight = word_embeddings.weight.detach().clone()
        del model
        return weight

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embedding(tokens.long()))

    def forward(self, upper, lower, lower_rc=None, upper_oh=None, lower_rc_oh=None, return_aux=False):
        u_feat = self._embed(upper)
        l_feat = self._embed(lower)

        parts = []
        aux = {}

        if self.use_cnn:
            cnn_u = self.cnn(u_feat.transpose(1, 2)).view(u_feat.size(0), -1)
            cnn_l = self.cnn(l_feat.transpose(1, 2)).view(l_feat.size(0), -1)
            parts.extend([cnn_u, cnn_l])

        if self.use_stem:
            if upper_oh is None or lower_rc_oh is None:
                raise ValueError("upper_oh and lower_rc_oh are required for embedding-only BSCAN stem branch.")
            bp = _bp_map(upper_oh, lower_rc_oh)
            row_max = bp.max(dim=-1).values
            col_max = bp.max(dim=-2).values
            x_stem = self.stem_cnn(bp.unsqueeze(1)).view(upper.size(0), -1)
            parts.extend([x_stem, row_max, col_max])
            if return_aux: aux['bp_map'] = bp

        if self.use_attn:
            attn_feat = u_feat
            for layer in self.attn_layers:
                attn_feat, _ = layer(attn_feat, l_feat, l_feat)
            parts.append(attn_feat.mean(dim=1))

        combined = torch.cat(parts, dim=1)
        logits = self.classifier(self.drop(combined))
        if return_aux:
            aux['combined'] = combined.detach().cpu()
            return logits, aux
        return logits

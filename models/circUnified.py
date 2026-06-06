import torch
import torch.nn as nn
import torch.nn.functional as F
from multimolecule import RnaErnieModel, RnaBertModel
from .mamba2 import Mamba2, Mamba2Config, RMSNorm as MambaRMSNorm
from .classifier import Classifier

class CircUnifiedBlock(nn.Module):
    """
    Final Research Architecture: Motif-Level Latent Interaction
    
    1. Direct 1D Path: Preserve high-res sequence motifs (Success factor of circCNNATT).
    2. Latent interaction Path: 
       - Downsize (L=200 -> L=50) to filter noise.
       - Contextualize (Mamba) to add global awareness.
       - Latent Attention (50x50) for motif-level interaction mapping.
    3. Hybrid Decision: Combine direct motifs, latent interactions, and global context.
    """
    def __init__(
        self,
        d_model=128,
        length_seq=200,
        n_heads=4,
        n_mamba_layers=2,
        dropout=0.1,
        use_pretrained=False,
        conv_out_channels=256,
        conv_kernel_size=11
    ):
        super().__init__()
        self.d_model = d_model
        
        # 1. Embedding Layer (Scratch)
        self.embeddings = nn.Embedding(26, d_model)
        nn.init.normal_(self.embeddings.weight, std=0.02)

        # 2. Strong 1D Path (Matches circCNNATT's resolution)
        self.direct_cnn = nn.Sequential(
            nn.Conv1d(d_model, conv_out_channels, kernel_size=conv_kernel_size, padding=conv_kernel_size//2),
            nn.BatchNorm1d(conv_out_channels), nn.ReLU(inplace=True), nn.MaxPool1d(4),
            nn.Conv1d(conv_out_channels, conv_out_channels, kernel_size=conv_kernel_size, padding=conv_kernel_size//2),
            nn.BatchNorm1d(conv_out_channels), nn.ReLU(inplace=True), nn.MaxPool1d(4)
        )
        self.cnn1d_feat_dim = (length_seq // 16) * conv_out_channels

        # 3. Latent Interaction Path (The Core Novelty)
        # 3-1. Downsizing (L=200 -> L=50)
        self.down_cnn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2), # L=100
            nn.BatchNorm1d(d_model), nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2), # L=50
            nn.BatchNorm1d(d_model), nn.ReLU(inplace=True)
        )
        
        # 3-2. Contextualizer (Mamba on L=50 tokens)
        self.mamba_config = Mamba2Config(
            d_model=d_model, n_layer=n_mamba_layers, d_state=64, d_conv=4, expand=2,
            headdim=d_model // n_heads, chunk_size=10 # 50 is divisible by 10
        )
        self.mamba_layers = nn.ModuleList([
            nn.ModuleDict({"mixer": Mamba2(self.mamba_config), "norm": MambaRMSNorm(d_model)}) 
            for _ in range(n_mamba_layers)
        ])

        # 3-3. Latent Attention (50x50)
        self.latent_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        
        # 4. Final Hybrid Classifier
        # Features: [Upper 1D] + [Lower 1D] + [Latent Interaction] + [Global Context]
        self.total_dim = (self.cnn1d_feat_dim * 2) + d_model + d_model
        self.classifier = Classifier(d_in=self.total_dim)

    def forward(self, upper_seq, lower_seq, lower_rc_seq, return_explain=False):
        u = self.embeddings(upper_seq)
        l = self.embeddings(lower_seq)
        l_rc = self.embeddings(lower_rc_seq)

        # Path A: Direct 1D motifs (High-Res)
        u_1d = self.direct_cnn(u.transpose(1, 2)).view(u.size(0), -1)
        l_1d = self.direct_cnn(l.transpose(1, 2)).view(l.size(0), -1)

        # Path B: Latent Motif Interaction
        # 1. Compress
        u_motif = self.down_cnn(u.transpose(1, 2)).transpose(1, 2)
        l_rc_motif = self.down_cnn(l_rc.transpose(1, 2)).transpose(1, 2)
        
        # 2. Mamba Context
        def run_mamba(x):
            for layer in self.mamba_layers:
                z, _ = layer["mixer"](layer["norm"](x))
                x = x + z
            return x
        u_m = run_mamba(u_motif)
        l_rc_m = run_mamba(l_rc_motif)
        
        # 3. Motif-to-Motif Attention
        latent_out, attn_map = self.latent_attn(u_m, l_rc_m, l_rc_m)
        
        # Summarize Path B
        inter_feat = latent_out.mean(dim=1)
        global_feat = u_m.mean(dim=1)

        # Final Fusion
        combined = torch.cat([u_1d, l_1d, inter_feat, global_feat], dim=1)
        logits = self.classifier(combined)

        if return_explain:
            return logits, {"attn_map": attn_map, "latent_motifs": u_motif}
        return logits

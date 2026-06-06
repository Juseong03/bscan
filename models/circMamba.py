import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .mamba2 import Mamba2


from torch.nn import MultiheadAttention
from .classifier import Classifier
import math

class ResidualFeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout_prob):
        super(ResidualFeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.norm(x + residual)
    
class MultiLayerAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout_prob):
        super(MultiLayerAttentionBlock, self).__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_prob),
                nn.LayerNorm(embed_dim),
                ResidualFeedForward(embed_dim, ff_dim, dropout_prob)
            ])
            for _ in range(num_layers)
        ])

    def forward(self, query, key, value):
        attn_output = query
        for attn, norm, ff in self.layers:
            attn_output, _ = attn(attn_output, key, value)
            attn_output = norm(attn_output)
            attn_output = ff(attn_output)
        return attn_output
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # Auto-adapt to input dimension if mismatch
        if x.size(-1) != self.weight.size(0):
            # Create new weight parameter with correct size
            new_weight = nn.Parameter(torch.ones(x.size(-1), device=x.device, dtype=x.dtype))
            # Copy existing values if possible
            min_dim = min(x.size(-1), self.weight.size(0))
            new_weight.data[:min_dim] = self.weight.data[:min_dim]
            self.weight = new_weight

        norm_x = x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return norm_x * self.weight


class Mamba2Block(nn.Module):
    def __init__(self, d_model, d_inner, d_head, d_conv, n_heads, d_state, n_groups, activation="silu", bias=False, conv_bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_head = d_head
        self.d_conv = d_conv
        self.n_heads = n_heads
        self.d_state = d_state
        self.n_groups = n_groups
        self.activation_fn = F.silu if activation == "silu" else F.silu

        # Ensure the input projection matches the input channels
        d_in_proj = self.d_inner  # Update this to match the actual input channels going into the conv layer
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias)

        # Convolution: in_channels must match the output of the in_proj layer
        conv_dim = self.d_inner + 2 * self.n_groups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=d_in_proj,  # Ensure this matches the output of in_proj
            out_channels=conv_dim,
            kernel_size=d_conv,
            groups=n_groups,  # Ensure this matches the correct number of groups
            bias=conv_bias,
            padding=d_conv - 1
        )

        # Learnable parameters
        self.A_log = nn.Parameter(torch.rand(self.n_heads))
        self.dt_bias = nn.Parameter(torch.rand(self.n_heads))
        self.D = nn.Parameter(torch.ones(self.n_heads))
        self.norm = RMSNorm(self.d_inner)
        # Auto-adapt output projection dimension
        self.out_proj = nn.Linear(self.d_inner, self.d_model)

    def forward(self, u, cache=None):
        batch, length, _ = u.shape
        if cache is not None and length == 1:
            return self.step(u, cache)

        zxbcdt = self.in_proj(u)
        A = -torch.exp(self.A_log)

        # Convolution: Ensure that the input dimensions match
        xBC = self.conv1d(zxbcdt.transpose(1, 2)).transpose(1, 2)
        x = self.activation_fn(xBC)

        # Output projection
        y = self.norm(x)
        out = self.out_proj(y)
        return out, cache

    def step(self, u, cache):
        h_cache, conv_cache = cache
        zxbcdt = self.in_proj(u.squeeze(1))
        z0, x0, z, xBC, dt = torch.split(zxbcdt, [self.d_inner] * 4 + [self.n_heads], dim=-1)

        # Convolution step
        conv_cache = torch.roll(conv_cache, shifts=-1, dims=-1)
        conv_cache[:, :, -1] = xBC
        xBC = torch.sum(conv_cache * self.conv1d.weight, dim=-1) + self.conv1d.bias
        xBC = self.activation_fn(xBC)

        # Update h_cache with new states
        A = -torch.exp(self.A_log)
        dt = F.softplus(dt + self.dt_bias)
        h_cache = h_cache * A.unsqueeze(0) + xBC.unsqueeze(2)

        y = h_cache.sum(dim=-1)
        out = self.norm(y)
        out = self.out_proj(out)
        return out.unsqueeze(1), (h_cache, conv_cache)


class ResidualBlock(nn.Module):
    def __init__(self, d_model, d_inner, d_head, d_conv, n_heads, d_state, n_groups, activation="silu", bias=False, conv_bias=True):
        super().__init__()
        self.mixer = Mamba2Block(d_model, d_inner, d_head, d_conv, n_heads, d_state, n_groups, activation, bias, conv_bias)
        self.norm = RMSNorm(d_model)

    def forward(self, x, cache=None):
        output, cache = self.mixer(self.norm(x), cache)
        return output + x, cache


class Mamba2(nn.Module):
    def __init__(
        self, 
        d_model, 
        n_layers, 
        d_head, 
        d_state, 
        expand_factor, 
        d_conv, 
        n_groups, 
        activation="silu", 
        bias=False, 
        conv_bias=True
    ):
        super().__init__()
        d_inner = expand_factor * d_model
        n_heads = d_inner // d_head
        self.layers = nn.ModuleList([ResidualBlock(d_model, d_inner, d_head, d_conv, n_heads, d_state, n_groups, activation, bias, conv_bias) for _ in range(n_layers)])

    def forward(self, x, caches=None):
        if caches is None:
            caches = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer(x, caches[i])

        return x, caches


class circMamba(nn.Module):
    def __init__(
        self, 
        d_model=128,               # Default embedding dimension
        n_layers=3,                # Number of layers in the Mamba block
        d_head=4,                  # Dimension of the heads for multi-head attention
        d_state=64,                # Default state dimension for Mamba blocks
        expand_factor=2,           # Expansion factor for the internal projection in Mamba
        d_conv=3,                  # Default convolution kernel size in Mamba blocks
        n_groups=1,                # Number of groups (for grouped convolution, default 1 means no grouping)
        activation="silu",         # Activation function
        bias=False,                # Whether to use bias in the Mamba block
        conv_bias=True,            # Whether to use bias in the convolution layers
        embedding_dim=64,          # Dimensionality of the embedding used before convolution
        conv_out_channels1=256,    # Output channels for the first convolutional layer
        conv_kernel_size1=11,      # Kernel size for the first convolutional layer
        conv_out_channels2=256,    # Output channels for the second convolutional layer
        conv_kernel_size2=11,      # Kernel size for the second convolutional layer
        input_dim=200,             # Input sequence length
        num_heads=4,               # Number of attention heads
        ff_dim=128,                # Feedforward dimension in the attention block
        num_attention_layers=3,    # Number of layers in the attention block
        dropout_prob=0.1           # Dropout probability
    ):
        super().__init__()

        # Mamba2-based components
        # Token ids come from `multimolecule`'s `RnaTokenizer` (RNAErnie/RNABert).
        # While A/C/G/T inputs often stay <= 9, ambiguous bases can exceed that (e.g. N/R/Y...).
        # Use a safe vocab size to avoid "index out of range" at runtime.
        self.embedding = nn.Embedding(26, d_model)
        self.mamba = Mamba2(d_model, n_layers, d_head, d_state, expand_factor, d_conv, n_groups, activation, bias, conv_bias)
        self.norm = RMSNorm(d_model)

        # CNN and Attention components (similar to CircCNNATT)
        self.to_proj = nn.Sequential(nn.Linear(d_model, embedding_dim), nn.GELU())

        # Shared convolutional layers for both upper and lower sequences
        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=conv_out_channels1, kernel_size=conv_kernel_size1, padding=(conv_kernel_size1 - 1) // 2),
            nn.BatchNorm1d(conv_out_channels1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4)
        )
        
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=conv_out_channels1, out_channels=conv_out_channels2, kernel_size=conv_kernel_size2, padding=(conv_kernel_size2 - 1) // 2),
            nn.BatchNorm1d(conv_out_channels2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4)
        )
        
        self.out_dim = input_dim // (4 * 4)  # Two pooling layers, pool size = 4

        # Multi-layer attention mechanism (Q = upper, K = lower reverse, V = lower)
        self.attention_block = MultiLayerAttentionBlock(
            embed_dim=embedding_dim, 
            num_heads=num_heads, 
            ff_dim=ff_dim, 
            num_layers=num_attention_layers, 
            dropout_prob=dropout_prob
        )

        # Fully connected layer after convolution and attention
        self.fc_input_dim = (conv_out_channels2 * self.out_dim * 2) + embedding_dim

        self.fc1 = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),  
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob)
        )

        self.fc_output = nn.Linear(128, 2)  # Output for binary classification

    def forward(self, upper_seq, lower_seq, lower_rc_seq, caches=None):
        # Step 1: Embedding
        upper_seq = self.embedding(upper_seq)
        lower_seq = self.embedding(lower_seq)
        lower_rc_seq = self.embedding(lower_rc_seq)

        # Step 2: Mamba2 processing
        out_upper, _ = self.mamba(self.norm(upper_seq))
        out_lower, _ = self.mamba(self.norm(lower_seq))
        out_lower_rc, _ = self.mamba(self.norm(lower_rc_seq))

        # Step 3: Convolutional layers
        upper_proj = self.to_proj(out_upper)
        lower_proj = self.to_proj(out_lower)
        lower_rc_proj = self.to_proj(out_lower_rc)

        upper_conv = self.conv_layer1(upper_proj.transpose(1, 2))
        upper_conv = self.conv_layer2(upper_conv)
        upper_conv = upper_conv.view(upper_conv.size(0), -1)

        lower_conv = self.conv_layer1(lower_proj.transpose(1, 2))
        lower_conv = self.conv_layer2(lower_conv)
        lower_conv = lower_conv.view(lower_conv.size(0), -1)

        # Step 4: Attention mechanism (upper_seq, lower_rc_seq, lower_seq)
        attention_output = self.attention_block(upper_proj, lower_rc_proj, lower_proj)
        attention_output = attention_output.mean(dim=1)  # Mean pooling of attention outputs

        # Step 5: Concatenate the results and pass through fully connected layers
        combined_output = torch.cat((upper_conv, lower_conv, attention_output), dim=1)
        out = self.fc1(combined_output)
        out = self.fc_output(out)

        return out

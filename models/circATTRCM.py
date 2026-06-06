import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from multimolecule import RnaTokenizer, RnaErnieModel, RnaBertModel
from .classifier import Classifier

class CircATTRCM(nn.Module):
    def __init__(
        self, 
        d_model=128,  
        length_seq=200, 
        n_heads=4,  
        ff_dim=128,  
        n_attn_layers=3,  
        dropout=0.1,
        pretrained_model='rnaernie',
        use_pretrained=False # Change default to False
    ):
        super(CircATTRCM, self).__init__()
        self.use_pretrained = use_pretrained

        if use_pretrained:
            # Load pretrained model (Ernie or BERT)
            if pretrained_model.lower() == 'rnaernie':
                pretrained = RnaErnieModel.from_pretrained('multimolecule/rnaernie')
            elif pretrained_model.lower() == 'rnabert':
                pretrained = RnaBertModel.from_pretrained('multimolecule/rnabert')
            else:
                raise ValueError(f'Invalid pretrained model: {pretrained_model}')
            
            # Freeze pretrained embedding parameters
            self.embeddings = pretrained.embeddings
            self.encoder = None # Not used in forward, remove to save GPU memory

            for param in self.embeddings.parameters():
                param.requires_grad = False

            # Project down to a smaller embedding size
            self.to_proj = nn.Sequential(nn.Linear(768, d_model), nn.GELU())
        else:
            self.embeddings = nn.Embedding(26, d_model)
            nn.init.normal_(self.embeddings.weight, std=0.02)
            self.to_proj = nn.Identity()

        # Add the attention block
        self.attention_block = MultiLayerAttentionBlock(
            embed_dim=d_model,
            num_heads=n_heads,
            ff_dim=ff_dim,
            num_layers=n_attn_layers,
            dropout_prob=dropout
        )

        # Define the convolutional block for the interaction map
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # Calculate the output dimension of the convolutional block
        self.fc_input_dim = self._calculate_conv_output_dim(length_seq)

        self.classifier = Classifier(d_in=self.fc_input_dim)

    def _calculate_conv_output_dim(self, input_dim):
        dummy_input = torch.zeros(1, 1, input_dim, input_dim)
        dummy_output = self.convblock(dummy_input)
        return int(dummy_output.numel())

    def forward(self, upper_seq, lower_seq, lower_rc_seq):
        upper_embedded = self.embeddings(upper_seq)
        lower_embedded = self.embeddings(lower_seq)
        lower_reverse_embedded = self.embeddings(lower_rc_seq)
        # upper_embedded = self.encoder(upper_embedded)['last_hidden_state']
        # lower_embedded = self.encoder(lower_embedded)['last_hidden_state']
        # lower_reverse_embedded = self.encoder(lower_reverse_embedded)['last_hidden_state']

        upper_embedded = self.to_proj(upper_embedded)
        lower_embedded = self.to_proj(lower_embedded)
        lower_reverse_embedded = self.to_proj(lower_reverse_embedded)

        upper_attn = self.attention_block(upper_embedded, lower_reverse_embedded, upper_embedded)
        lower_attn = self.attention_block(upper_embedded, lower_reverse_embedded, lower_embedded)

        interaction_map = torch.einsum('bld,bmd->blm', upper_attn, lower_attn)  # Interaction map (B, L, L)
        interaction_map = interaction_map.unsqueeze(1)  # Add a channel dimension for Conv2d (B, 1, L, L)

        conv_output = self.convblock(interaction_map)
        conv_output = conv_output.view(conv_output.size(0), -1)

        out = self.classifier(conv_output)

        return out



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
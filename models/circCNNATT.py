import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from multimolecule import RnaErnieModel
from multimolecule import RnaBertModel
from .classifier import Classifier

class CircCNNATT(nn.Module):
    def __init__(
        self, 
        d_model=128,
        conv_out_channels1=256,  
        conv_kernel_size1=11,    
        conv_out_channels2=256, 
        conv_kernel_size2=11, 
        length_seq=200, 
        n_heads=4,  
        ff_dim=128,
        n_attntions=3,
        dropout_prob=0.1,
        pretrained_model='rnaernie',
        use_pretrained=False # Change default to False
    ):
        super(CircCNNATT, self).__init__()
        self.use_pretrained = use_pretrained

        if use_pretrained:
            if pretrained_model.lower() == 'rnaernie':
                pretrained = RnaErnieModel.from_pretrained('multimolecule/rnaernie')
            elif pretrained_model.lower() == 'rnabert':
                pretrained = RnaBertModel.from_pretrained('multimolecule/rnabert')
            else:
                raise ValueError(f'Invalid pretrained model: {pretrained_model}')
            self.embeddings = pretrained.embeddings
            self.encoder = None # Remove encoder to save memory

            for param in self.embeddings.parameters():
                param.requires_grad = False
            self.to_proj = nn.Sequential(nn.Linear(768, d_model), nn.GELU())
        else:
            # Training from scratch with simple embedding matrix
            self.embeddings = nn.Embedding(26, d_model)
            nn.init.normal_(self.embeddings.weight, std=0.02)
            self.to_proj = nn.Identity()

        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=conv_out_channels1, kernel_size=conv_kernel_size1, padding=(conv_kernel_size1 - 1) // 2),
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
        self.out_dim = length_seq // (4 * 4)

        self.attention_block = MultiLayerAttentionBlock(
            embed_dim=d_model, 
            num_heads=n_heads, 
            ff_dim=ff_dim, 
            num_layers=n_attntions, 
            dropout_prob=dropout_prob
        )
        self.fc_input_dim = (conv_out_channels2 * self.out_dim * 2) + d_model
        # self.fc_input_dim = (d_model * 2) + d_model

        self.classifier = Classifier(d_in=self.fc_input_dim)

    def forward(self, upper_seq, lower_seq, lower_rc_seq):
        upper_embedded = self.embeddings(upper_seq)
        lower_embedded = self.embeddings(lower_seq)
        lower_reverse_embedded = self.embeddings(lower_rc_seq)

        upper_embedded = self.to_proj(upper_embedded)
        lower_embedded = self.to_proj(lower_embedded)
        lower_reverse_embedded = self.to_proj(lower_reverse_embedded)

        upper_conv = self.conv_layer1(upper_embedded.transpose(1, 2))
        upper_conv = self.conv_layer2(upper_conv)
        upper_conv = upper_conv.view(upper_conv.size(0), -1)

        lower_conv = self.conv_layer1(lower_embedded.transpose(1, 2))  
        lower_conv = self.conv_layer2(lower_conv)
        lower_conv = lower_conv.view(lower_conv.size(0), -1)

        attention_output = self.attention_block(upper_embedded, lower_reverse_embedded, lower_embedded)

        out = torch.cat((upper_conv, lower_conv, attention_output[:, 0, :]), dim=1)
        out = self.classifier(out)

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
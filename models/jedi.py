from .classifier import Classifier
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W = nn.Linear(units, units)  # Weight matrix for attention
        self.V = nn.Linear(units, 1)      # Projection for attention scores

    def forward(self, values):
        score = self.V(torch.tanh(self.W(values)))  # (batch_size, sequence_length, 1)
        attention_weights = F.softmax(score, dim=1)  # (batch_size, sequence_length, 1)
        context_vector = attention_weights * values  # (batch_size, sequence_length, units)
        context_vector = torch.sum(context_vector, dim=1)  # Sum over sequence length
        return context_vector, attention_weights


class JEDI(nn.Module):
    def __init__(
            self, 
            K=6, 
            L=100, 
            max_len=50, 
            d_model=128, 
            d_rnn=128, 
            d_attn=128, 
            d_hidden=256, 
            l2_reg=0.01
        ):
        super(JEDI, self).__init__()

        self.K = K
        self.L = L
        self.max_len = max_len
        self.emb_dim = d_model
        self.rnn_dim = d_rnn
        self.att_dim = d_attn
        self.hidden_dim = d_hidden
        self.l2_reg = l2_reg

        # K-mer embedding layer
        self.to_embeddings = nn.Embedding(5 ** K, d_model, padding_idx=0)

        # Bidirectional GRUs for donor and acceptor sequences
        self.rnn_a = nn.GRU(d_model, d_rnn, batch_first=True, bidirectional=True)
        self.rnn_d = nn.GRU(d_model, d_rnn, batch_first=True, bidirectional=True)

        # Project GRU output to attention dimension
        self.projection = nn.Linear(d_rnn * 2, d_attn)

        # Bahdanau Attention for acceptor and donor
        self.seq_att_a = BahdanauAttention(d_attn)
        self.seq_att_d = BahdanauAttention(d_attn)

        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_attn, num_heads=8, batch_first=True)

        # Final Bahdanau Attention
        self.final_att_a = BahdanauAttention(d_attn)
        self.final_att_d = BahdanauAttention(d_attn)

        # Fully connected layers
        self.classifier = Classifier(d_in=d_attn*2)

    def forward(self, xa, xd):
        emb_a = self.to_embeddings(xa)  # (batch_size, seq_length, emb_dim)
        emb_d = self.to_embeddings(xd)

        # GRU layers
        vectors_a, _ = self.rnn_a(emb_a)  # (batch_size, seq_length, 2 * rnn_dim)
        vectors_d, _ = self.rnn_d(emb_d)

        # Projection to attention dimension
        vectors_a = self.projection(vectors_a)  # (batch_size, seq_length, att_dim)
        vectors_d = self.projection(vectors_d)

        # Bahdanau attention
        context_a, _ = self.seq_att_a(vectors_a)  # (batch_size, att_dim)
        context_d, _ = self.seq_att_d(vectors_d)  # (batch_size, att_dim)

        # Cross-attention
        vectors_a_by_d, _ = self.cross_attention(context_a.unsqueeze(1), context_d.unsqueeze(1), context_d.unsqueeze(1))
        vectors_d_by_a, _ = self.cross_attention(context_d.unsqueeze(1), context_a.unsqueeze(1), context_a.unsqueeze(1))

        # Final attention mechanism
        final_features = torch.cat([self.final_att_a(vectors_a_by_d)[0], self.final_att_d(vectors_d_by_a)[0]], dim=1)

        # Fully connected layers
        pred = self.classifier(final_features)

        return pred

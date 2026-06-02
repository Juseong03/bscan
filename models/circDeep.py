import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .classifier import Classifier

class CircDeep(nn.Module):
    def __init__(
            self, 
            kmer=3, 
            d_model=128, 
            embedding_matrix=None, 
            num_filters=100, 
            lstm_units=100
        ):
        super(CircDeep, self).__init__()
        
        # Embedding layer with pre-trained Word2Vec embeddings
        self.embedding = nn.Embedding(4**kmer, d_model)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
            self.embedding.weight.requires_grad = True  # Make the embeddings trainable

        # First Convolutional block
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=num_filters, kernel_size=7, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Second Convolutional block
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=1, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bidirectional LSTM
        self.bilstm = nn.LSTM(input_size=num_filters, hidden_size=lstm_units, num_layers=1, bidirectional=True, batch_first=True)

        # Fully connected layers
        self.classifier = Classifier(d_in=lstm_units * 2)  # LSTM is bidirectional, hence *2

    def forward(self, x):
        # Embedding layer
        x = self.embedding(x)  # Shape: (batch_size, max_len, embedding_dim)
        x = x.permute(0, 2, 1)  # Permute to (batch_size, embedding_dim, max_len) for Conv1D
        
        # First convolutional block
        x = F.relu(self.conv1(x))  # (batch_size, num_filters, new_len)
        x = self.pool1(x)  # MaxPooling layer
        
        # Second convolutional block
        x = F.relu(self.conv2(x))  # (batch_size, num_filters, reduced_len)
        x = self.pool2(x)
        
        # Transpose for LSTM input: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # Bidirectional LSTM
        x, (h_n, c_n) = self.bilstm(x)
        
        # Only use the final LSTM output for classification
        x = x[:, -1, :]  # Take the output of the last time step
        
        # Fully connected layers
        x = self.classifier(x)
        
        return x
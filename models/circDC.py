"""
CircDC integration note
----------------------
The original CircDC paper focuses on circRNA vs lncRNA classification and uses multiple handcrafted
feature encodings plus a CNN+BiLSTM backbone (with interpretability analysis).

This repository's main task is BS vs LS (exon-pair) classification using junction/flanking sequences.
To integrate a "CircDC-like" baseline into this BS/LS pipeline, we implement a practical CNN+BiLSTM
model that consumes the same inputs as the other baselines in this repo: upper/lower one-hot tensors
of shape (B, 4, L) where L = 2*junction_bps (default 200).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import Classifier


class _SeqEncoder(nn.Module):
    """Shared CNN + BiLSTM encoder for a single one-hot sequence (B, 4, L)."""

    def __init__(
        self,
        in_channels: int = 4,
        conv1_out_channels: int = 100,
        conv1_kernel_size: int = 7,
        conv2_out_channels: int = 100,
        conv2_kernel_size: int = 1,
        lstm_hidden_size: int = 100,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=conv1_out_channels,
            kernel_size=conv1_kernel_size,
            padding=(conv1_kernel_size - 1) // 2,
        )
        self.maxpool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv2 = nn.Conv1d(
            in_channels=conv1_out_channels,
            out_channels=conv2_out_channels,
            kernel_size=conv2_kernel_size,
            padding=(conv2_kernel_size - 1) // 2,
        )
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm = nn.LSTM(
            input_size=conv2_out_channels,
            hidden_size=lstm_hidden_size,
            bidirectional=True,
            batch_first=True,
        )

        self.out_dim = lstm_hidden_size * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, L)
        x = self.dropout(x)
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.dropout(x)

        # BiLSTM expects (B, T, C)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.dropout(x)

        # Pool across time -> (B, 2H)
        x = x.mean(dim=1)
        return x


class CircDC(nn.Module):
    """
    BS/LS-integrated CircDC-like model:
    - encodes upper & lower junction sequences with a shared CNN+BiLSTM
    - concatenates representations and classifies (2 classes)
    """

    def __init__(
        self,
        in_channels: int = 4,
        conv1_out_channels: int = 100,
        conv1_kernel_size: int = 7,
        conv2_out_channels: int = 100,
        conv2_kernel_size: int = 1,
        lstm_hidden_size: int = 100,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.encoder = _SeqEncoder(
            in_channels=in_channels,
            conv1_out_channels=conv1_out_channels,
            conv1_kernel_size=conv1_kernel_size,
            conv2_out_channels=conv2_out_channels,
            conv2_kernel_size=conv2_kernel_size,
            lstm_hidden_size=lstm_hidden_size,
            dropout_rate=dropout_rate,
        )

        self.classifier = Classifier(d_in=self.encoder.out_dim * 2)

    def forward(self, upper_seq: torch.Tensor, lower_seq: torch.Tensor) -> torch.Tensor:
        # upper_seq/lower_seq: (B, 4, L)
        u = self.encoder(upper_seq)
        l = self.encoder(lower_seq)
        x = torch.cat([u, l], dim=1)
        return self.classifier(x)
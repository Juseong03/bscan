import torch
import torch.nn as nn
import torch.nn.functional as F
from .classifier import Classifier

class CircCNN(nn.Module):
    def __init__(
        self, 
        junction_bps=100, 
        conv1_out_channels=256, 
        conv2_out_channels=128, 
        kernel_size1=12, 
        kernel_size2=30, 
        stride1=1, 
        stride2=2, 
        pool_kernel_size=5, 
        pool_stride=5, 
        dropout1=0.0, 
        dropout2=0.3,
    ):
        super(CircCNN, self).__init__()

        length_seq = junction_bps * 2

        self.upper_conv1 = nn.Conv1d(4, conv1_out_channels, kernel_size=kernel_size1, stride=stride1, padding=0)
        self.upper_conv2 = nn.Conv1d(conv1_out_channels, conv2_out_channels, kernel_size=kernel_size2, stride=stride2, padding=kernel_size2 // 2)
        self.upper_dropout1 = nn.Dropout(dropout1)
        self.upper_dropout2 = nn.Dropout(dropout2)
        self.upper_pool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)

        self.lower_conv1 = nn.Conv1d(4, conv1_out_channels, kernel_size=kernel_size1, stride=stride1, padding=0)
        self.lower_conv2 = nn.Conv1d(conv1_out_channels, conv2_out_channels, kernel_size=kernel_size2, stride=stride2, padding=kernel_size2 // 2)
        self.lower_dropout1 = nn.Dropout(dropout1)
        self.lower_dropout2 = nn.Dropout(dropout2)
        self.lower_pool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)

        # Dynamically calculate flattened size
        with torch.no_grad():
            dummy_input = torch.randn(1, 4, length_seq)
            dummy_out = self.upper_pool(F.relu(self.upper_conv2(F.relu(self.upper_conv1(dummy_input)))))
            flattened_size = dummy_out.flatten(start_dim=1).size(1)

        self.classifier = Classifier(d_in=flattened_size * 2)

    def forward(self, x1, x2):
        x1 = F.relu(self.upper_conv1(x1))
        x1 = self.upper_dropout1(x1)
        x1 = F.relu(self.upper_conv2(x1))
        x1 = self.upper_dropout2(x1)
        x1 = self.upper_pool(x1)
        x1 = x1.flatten(start_dim=1)

        x2 = F.relu(self.lower_conv1(x2))
        x2 = self.lower_dropout1(x2)
        x2 = F.relu(self.lower_conv2(x2))
        x2 = self.lower_dropout2(x2)
        x2 = self.lower_pool(x2)
        x2 = x2.flatten(start_dim=1)

        # Concatenate the features
        x = torch.cat((x1, x2), dim=1) 

        # Fully connected layers
        x = self.classifier(x)
        
        return x

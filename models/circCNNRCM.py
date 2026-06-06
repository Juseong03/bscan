import torch
import torch.nn as nn
import torch.nn.functional as F
from .classifier import Classifier

class RCMBlock(nn.Module):
    '''
    Model to process the RCM score distribution of the flanking introns.
    '''
    def __init__(self, in_channels=5, out_channel1=512, out_channel2=128, kernel_size=5, stride=5):
        super(RCMBlock, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channel1, 
                               kernel_size=kernel_size, stride=stride, padding=0)
        self.conv1_bn = nn.BatchNorm1d(out_channel1)  # Batch Normalization for the first layer

        # Second Convolutional Layer
        self.conv2 = nn.Conv1d(in_channels=out_channel1, out_channels=out_channel2, 
                               kernel_size=kernel_size, stride=stride, padding=0)
        self.conv2_bn = nn.BatchNorm1d(out_channel2)  # Batch Normalization for the second layer

    def forward(self, x):
        # Pass through the first convolutional layer and apply ReLU activation
        x = F.relu(self.conv1_bn(self.conv1(x)))
        
        # Pass through the second convolutional layer and apply ReLU activation
        x = F.relu(self.conv2_bn(self.conv2(x)))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        return x

class CircCNNRCM(nn.Module):
    '''
    Model to process and concatenate RCM score distributions from flanking, upper, and lower introns.
    '''
    def __init__(
        self,
        in_channels=5,
        out_channel1=512,
        out_channel2=128,
        kernel_size=5,
        stride=5,
        # Number of RCM "feature maps" concatenated along the length dimension.
        # In the CircCNNs paper setting, this is typically len(flanking_list)*len(kmer_list) = 10*5 = 50.
        n_rcm_features=50,
    ):
        super(CircCNNRCM, self).__init__()

        # CNN models for RCM scores
        self.rcm_flanking = RCMBlock(in_channels, out_channel1, out_channel2, kernel_size, stride)
        self.rcm_upper = RCMBlock(in_channels, out_channel1, out_channel2, kernel_size, stride)
        self.rcm_lower = RCMBlock(in_channels, out_channel1, out_channel2, kernel_size, stride)

        if n_rcm_features < kernel_size:
            raise ValueError(f"n_rcm_features ({n_rcm_features}) must be >= kernel_size ({kernel_size}).")

        # Input length is 5 * n_rcm_features (5 bins per axis; concatenate matrices along columns).
        # After conv1: length -> n_rcm_features
        # After conv2: length -> (n_rcm_features - kernel_size) // stride + 1
        conv2_out_len = (n_rcm_features - kernel_size) // stride + 1

        # Calculate total input dimension for the fully connected layers
        self.fc1_input_dim = (conv2_out_len * out_channel2) * 3

        # First fully connected layer
        self.classifier = Classifier(self.fc1_input_dim)

    def forward(self, rcm_flanking, rcm_upper, rcm_lower):
        # Process each RCM score input through the respective CNN models
        x_flanking = self.rcm_flanking(rcm_flanking)
        x_upper = self.rcm_upper(rcm_upper)
        x_lower = self.rcm_lower(rcm_lower)

        # Concatenate the outputs from the three models
        x = torch.cat((x_flanking, x_upper, x_lower), dim=1)

        # Pass the concatenated features through the fully connected layers
        out = self.classifier(x)
        return out

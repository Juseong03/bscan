import torch
import torch.nn as nn
import torch.nn.functional as F
from .classifier import Classifier

class UpperLowerCNN(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, kernel_size1, kernel_size2, input_dim):
        super(UpperLowerCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels1, kernel_size=kernel_size1, stride=1, padding=(kernel_size1 - 1) // 2)
        self.conv1_bn = nn.BatchNorm1d(out_channels1)
        self.maxpool1 = nn.MaxPool1d(5)
        
        self.conv2 = nn.Conv1d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=kernel_size2, stride=1, padding=(kernel_size2 - 1) // 2)
        self.conv2_bn = nn.BatchNorm1d(out_channels2)
        self.maxpool2 = nn.MaxPool1d(10)

        # Calculate output dimension after convolutions and pooling
        self.fc_input_dim = out_channels2 * (input_dim // (5 * 10))

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.maxpool1(x)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.maxpool2(x)
        return x.view(x.size(0), -1)  # Flatten


class RCNN(nn.Module):
    def __init__(self, in_channels, flanking_channels1, flanking_channels2, kernel_size, n_rcm_features, stride=5):
        super(RCNN, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.n_rcm_features = n_rcm_features

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=flanking_channels1,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        self.conv1_bn = nn.BatchNorm1d(flanking_channels1)
        self.conv2 = nn.Conv1d(
            in_channels=flanking_channels1,
            out_channels=flanking_channels2,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        self.conv2_bn = nn.BatchNorm1d(flanking_channels2)

        if n_rcm_features < kernel_size:
            raise ValueError(f"n_rcm_features ({n_rcm_features}) must be >= kernel_size ({kernel_size}).")

        # Input length is 5 * n_rcm_features.
        # After conv1: length -> n_rcm_features
        # After conv2: length -> (n_rcm_features - kernel_size)//stride + 1
        conv2_out_len = (n_rcm_features - kernel_size) // stride + 1
        self.fc_input_dim = flanking_channels2 * conv2_out_len

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        return x.view(x.size(0), -1)  # Flatten
    

class CircCNNtri(nn.Module):
    def __init__(self, 
                 in_channels=4, 
                 channels1=256, 
                 kernel_size1=15, 
                 channels2=128, 
                 kernel_size2=21, 
                 rcm_channels1=128, 
                 rcm_channels2=64, 
                 rcm_kernel_size=5,
                 length_seq=200, 
                 # Number of RCM feature maps (= len(flanking_list)*len(kmer_list) in preprocessing).
                 # Paper default is often 50 (10 flanking lengths * 5 kmer sizes).
                 n_rcm_features=50,
                 **kwargs):
        super(CircCNNtri, self).__init__()

        self.upper_convs = UpperLowerCNN(in_channels, channels1, channels2, kernel_size1, kernel_size2, length_seq)
        self.lower_convs = UpperLowerCNN(in_channels, channels1, channels2, kernel_size1, kernel_size2, length_seq)

        fc_input_dim_upper_lower = self.upper_convs.fc_input_dim + self.lower_convs.fc_input_dim

        self.rcm_flanking = RCNN(5, rcm_channels1, rcm_channels2, rcm_kernel_size, n_rcm_features=n_rcm_features)
        self.rcm_upper = RCNN(5, rcm_channels1, rcm_channels2, rcm_kernel_size, n_rcm_features=n_rcm_features)
        self.rcm_lower = RCNN(5, rcm_channels1, rcm_channels2, rcm_kernel_size, n_rcm_features=n_rcm_features)

        fc_input_dim_rcm = (self.rcm_flanking.fc_input_dim +
                             self.rcm_upper.fc_input_dim +
                             self.rcm_lower.fc_input_dim)

        # Combine dimensions
        self.fc_input_dim = fc_input_dim_upper_lower + fc_input_dim_rcm

        self.classifier = Classifier(d_in=self.fc_input_dim)

    def forward(self, seq_upper_feature, seq_lower_feature, rcm_flanking, rcm_upper, rcm_lower):
        # Process upper and lower sequences
        x_upper = self.upper_convs(seq_upper_feature)
        x_lower = self.lower_convs(seq_lower_feature)

        # Combine upper and lower features
        x_seq = torch.cat((x_upper, x_lower), dim=1)

        # Process RCM features
        x_rcm_flanking = self.rcm_flanking(rcm_flanking)
        x_rcm_upper = self.rcm_upper(rcm_upper)
        x_rcm_lower = self.rcm_lower(rcm_lower)

        # Combine RCM features
        x_rcm = torch.cat((x_rcm_flanking, x_rcm_upper, x_rcm_lower), dim=1)
        
        # Combine sequence and RCM features
        x = torch.cat((x_seq, x_rcm), dim=1)

        # Fully connected layers
        out = self.classifier(x)
        return out

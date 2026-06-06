from .classifier import Classifier
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Module(nn.Module):
    """
    This class defines a generic CNN block to process sequences.
    It will be used for both upper and lower sequences.
    """

    def __init__(self, in_channels, out_channel1, out_channel2, kernel_size1, kernel_size2, maxpool1, maxpool2):
        super(CNN_Module, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channel1, 
                               kernel_size=kernel_size1, padding=(kernel_size1 - 1) // 2)
        self.conv1_bn = nn.BatchNorm1d(out_channel1)
        self.maxpool1 = maxpool1

        # Second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=out_channel1, out_channels=out_channel2, 
                               kernel_size=kernel_size2, padding=(kernel_size2 - 1) // 2)
        self.conv2_bn = nn.BatchNorm1d(out_channel2)
        self.maxpool2 = maxpool2

        self.conv2_out_dim = 200 // (self.maxpool1 * self.maxpool2)

    def forward(self, x):
        out = torch.relu(self.conv1_bn(self.conv1(x)))
        out = F.max_pool1d(out, self.maxpool1)
        out = torch.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool1d(out, self.maxpool2)
        out = out.view(out.size(0), -1)  # Flatten for fully connected layers
        return out


class RCM_Module(nn.Module):
    """
    This class defines a CNN to process RCM score distributions (Flanking, Upper, Lower).
    """

    def __init__(self, in_channels, out_channel1, out_channel2, kernel_size):
        super(RCM_Module, self).__init__()

        # First convolutional layer for RCM
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channel1, kernel_size=kernel_size, stride=5)
        self.conv1_bn = nn.BatchNorm1d(out_channel1)

        # Second convolutional layer for RCM
        self.conv2 = nn.Conv1d(in_channels=out_channel1, out_channels=out_channel2, kernel_size=kernel_size, stride=5)
        self.conv2_bn = nn.BatchNorm1d(out_channel2)

        self.conv2_out_dim = 10  # Final output dimension after pooling

    def forward(self, x):
        out = torch.relu(self.conv1_bn(self.conv1(x)))
        out = torch.relu(self.conv2_bn(self.conv2(out)))
        out = out.view(out.size(0), -1)  # Flatten for fully connected layers
        return out


class CircRCM(nn.Module):
    """
    This class defines the overall combined model that integrates sequence processing and RCM features.
    """

    def __init__(self):
        super(CircRCM, self).__init__()

        # CNN modules for upper and lower sequence features
        self.cnn_upper = CNN_Module(in_channels=4, out_channel1=512, out_channel2=512, 
                                    kernel_size1=15, kernel_size2=21, maxpool1=5, maxpool2=10)
        self.cnn_lower = CNN_Module(in_channels=4, out_channel1=256, out_channel2=512, 
                                    kernel_size1=13, kernel_size2=21, maxpool1=5, maxpool2=10)

        # CNN modules for RCM features (Flanking, Upper, Lower)
        self.rcm_flanking = RCM_Module(in_channels=5, out_channel1=128, out_channel2=128, kernel_size=5)
        self.rcm_upper = RCM_Module(in_channels=5, out_channel1=512, out_channel2=256, kernel_size=5)
        self.rcm_lower = RCM_Module(in_channels=5, out_channel1=512, out_channel2=512, kernel_size=5)

        # Fully connected layers to combine features
        upper_lower_in_features = self.cnn_upper.conv2_out_dim * 512 + self.cnn_lower.conv2_out_dim * 512
        self.upper_lower_fc1 = nn.Linear(upper_lower_in_features, 512)
        self.upper_lower_fc2 = nn.Linear(512, 4)

        rcm_in_features = (self.rcm_flanking.conv2_out_dim * 128) + \
                          (self.rcm_upper.conv2_out_dim * 256) + \
                          (self.rcm_lower.conv2_out_dim * 512)
        self.rcm_fc1 = nn.Linear(rcm_in_features, 512)
        self.rcm_fc2 = nn.Linear(512, 8)

        # Final fully connected layers
        combined_in_features = 4 + 8
        self.classifier = Classifier(combined_in_features)

    def forward(self, seq_upper, seq_lower, rcm_flanking, rcm_upper, rcm_lower):
        # Process sequence data
        upper_out = self.cnn_upper(seq_upper)
        lower_out = self.cnn_lower(seq_lower)
        upper_lower_out = torch.cat((upper_out, lower_out), dim=1)
        upper_lower_out = torch.relu(self.upper_lower_fc1(upper_lower_out))
        upper_lower_out = torch.relu(self.upper_lower_fc2(upper_lower_out))

        # Process RCM data
        rcm_flanking_out = self.rcm_flanking(rcm_flanking)
        rcm_upper_out = self.rcm_upper(rcm_upper)
        rcm_lower_out = self.rcm_lower(rcm_lower)
        rcm_out = torch.cat((rcm_flanking_out, rcm_upper_out, rcm_lower_out), dim=1)
        rcm_out = torch.relu(self.rcm_fc1(rcm_out))
        rcm_out = torch.relu(self.rcm_fc2(rcm_out))

        # Combine sequence and RCM features
        combined_out = torch.cat((upper_lower_out, rcm_out), dim=1)
        # Final output layer
        out = self.classifier(combined_out)
        return out

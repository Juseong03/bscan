import torch
import torch.nn as nn
from .classifier import Classifier

class CircCNNDoubleShare(nn.Module):
    def __init__(
        self, 
        in_channels=4, 
        out_channels1=512, 
        kernel_size1=15, 
        out_channels2=512, 
        kernel_size2=21, 
        length_seq=200, 
        dropout=0.3,
        activation_fn=nn.ReLU,  # Option to use different activation functions
        use_residual=False  # Option to use residual connections
    ):
        super(CircCNNDoubleShare, self).__init__()
        self.use_residual = use_residual
        
        # Helper function to define convolutional layers
        def conv_layer(in_channels, out_channels, kernel_size, pool_size=5, activation_fn=activation_fn):
            return nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                          stride=1, padding=(kernel_size - 1) // 2),
                nn.BatchNorm1d(out_channels),
                activation_fn(inplace=True),
                nn.MaxPool1d(pool_size)
            )
        
        self.conv1 = conv_layer(in_channels, out_channels1, kernel_size1)
        self.conv2 = conv_layer(out_channels1, out_channels2, kernel_size2)
        self.out_dim = length_seq // (5 * 5)  # Two pooling layers, pool size = 5

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc_input_dim = (out_channels2 + out_channels2)  # After global average pooling, only channels remain

        self.classifier = Classifier(d_in=self.fc_input_dim)

    def forward(self, seq_upper_feature, seq_lower_feature):
        # Upper sequence path
        out_upper = self.conv1(seq_upper_feature)
        if self.use_residual:
            out_upper_res = out_upper
        out_upper = self.conv2(out_upper)
        if self.use_residual:
            out_upper += out_upper_res  # Residual connection

        # Global average pooling
        out_upper = self.global_avg_pool(out_upper).squeeze(-1)  # Remove last dimension

        # Lower sequence path
        out_lower = self.conv1(seq_lower_feature)
        if self.use_residual:
            out_lower_res = out_lower
        out_lower = self.conv2(out_lower)
        if self.use_residual:
            out_lower += out_lower_res  # Residual connection

        # Global average pooling
        out_lower = self.global_avg_pool(out_lower).squeeze(-1)

        # Concatenate upper and lower features
        out = torch.cat((out_upper, out_lower), dim=1)

        # Forward through fully connected layers
        out = self.classifier(out)
        return out

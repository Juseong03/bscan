import torch
import torch.nn as nn
from .classifier import Classifier

class CircCNNDouble(nn.Module):
    def __init__(
        self, 
        in_channels=4, 
        upper_out_channels1=512, 
        upper_kernel_size1=15, 
        upper_out_channels2=512, 
        upper_kernel_size2=21, 
        lower_out_channels1=256, 
        lower_kernel_size1=13, 
        lower_out_channels2=512, 
        lower_kernel_size2=21,
        length_seq=200, 
        dropout=0.3,
        activation_fn=nn.ReLU,
        use_residual=False,
        **kwargs
    ):
        super(CircCNNDouble, self).__init__()
        self.use_residual = use_residual
        
        def conv_layer(in_channels, out_channels, kernel_size, pool_size=5, activation_fn=activation_fn):
            return nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                          stride=1, padding=(kernel_size - 1) // 2),
                nn.BatchNorm1d(out_channels),
                activation_fn(inplace=True),
                nn.MaxPool1d(pool_size)
            )
        
        self.upper_conv1 = conv_layer(in_channels, upper_out_channels1, upper_kernel_size1)
        self.upper_conv2 = conv_layer(upper_out_channels1, upper_out_channels2, upper_kernel_size2)
        self.upper_out_dim = length_seq // (5 * 5)

        self.lower_conv1 = conv_layer(in_channels, lower_out_channels1, lower_kernel_size1)
        self.lower_conv2 = conv_layer(lower_out_channels1, lower_out_channels2, lower_kernel_size2)
        self.lower_out_dim = length_seq // (5 * 5)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc_input_dim = (upper_out_channels2 + lower_out_channels2)

        self.classifier = Classifier(d_in=self.fc_input_dim)

    def forward(self, seq_upper_feature, seq_lower_feature):
        out_upper = self.upper_conv1(seq_upper_feature)
        if self.use_residual:
            out_upper_res = out_upper
        out_upper = self.upper_conv2(out_upper)
        if self.use_residual:
            out_upper += out_upper_res  # Residual connection

        out_upper = self.global_avg_pool(out_upper).squeeze(-1)  # Remove last dimension

        out_lower = self.lower_conv1(seq_lower_feature)
        if self.use_residual:
            out_lower_res = out_lower
        out_lower = self.lower_conv2(out_lower)
        if self.use_residual:
            out_lower += out_lower_res  # Residual connection

        out_lower = self.global_avg_pool(out_lower).squeeze(-1)

        out = torch.cat((out_upper, out_lower), dim=1)

        out = self.classifier(out)
        return out

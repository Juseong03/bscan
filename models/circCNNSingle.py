import torch
import torch.nn as nn
import torch.nn.functional as F
from .classifier import Classifier

class CircCNNSingle(nn.Module):
    def __init__(
        self, 
        in_channels=4, 
        upper_out_channels1=512, 
        upper_kernel_size1=15, 
        upper_out_channels2=512, 
        upper_kernel_size2=21, 
        # NOTE: for this project’s default preprocessing (`is_concat=True`),
        # input length is 4 * junction_bps (upper_intron+upper_exon+lower_exon+lower_intron).
        # With default junction_bps=100, that is 400.
        upper_input_dim=400,
        dropout_prob=0.3,
        pool_kernel_size=5,
        activation_fn=nn.ReLU,
        use_residual=False
    ):
        """
        Improved Single input CNN model for circRNA classification.
        :param in_channels: Number of input channels for CNN layers
        :param upper_out_channels1: Number of output channels for the first upper CNN layer
        :param upper_kernel_size1: Kernel size for the first upper CNN layer
        :param upper_out_channels2: Number of output channels for the second upper CNN layer
        :param upper_kernel_size2: Kernel size for the second upper CNN layer
        :param upper_input_dim: Input dimension for the upper sequence
        :param dropout_prob: Dropout probability for regularization
        :param pool_kernel_size: Pooling kernel size
        :param activation_fn: Activation function (default: ReLU)
        :param use_residual: Use residual connections (default: False)
        """
        super(CircCNNSingle, self).__init__()
        self.use_residual = use_residual

        # Helper function to define convolutional layers
        def conv_layer(in_channels, out_channels, kernel_size, pool_kernel_size=pool_kernel_size):
            return nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                          stride=1, padding=(kernel_size - 1) // 2),
                nn.BatchNorm1d(out_channels),
                activation_fn(inplace=True),
                nn.MaxPool1d(pool_kernel_size)
            )
        
        # Convolutional layers for upper sequence
        self.conv1 = conv_layer(in_channels, upper_out_channels1, upper_kernel_size1)
        self.conv2 = conv_layer(upper_out_channels1, upper_out_channels2, upper_kernel_size2)

        # If residual is enabled, downsample conv1 output to match conv2 output length.
        # conv2 includes its own maxpool; conv1 output needs an extra pool to match lengths.
        self.residual_pool = nn.MaxPool1d(pool_kernel_size) if use_residual else None

        # Infer the classifier input dim from a dummy forward pass to avoid shape mismatches
        # when `upper_input_dim` (i.e. junction_bps) changes.
        self.fc_input_dim = self._infer_fc_input_dim(in_channels=in_channels, input_len=upper_input_dim)
               
        # Fully connected layers
        self.classifier = Classifier(self.fc_input_dim)

    def _infer_fc_input_dim(self, in_channels: int, input_len: int) -> int:
        was_training = self.training
        try:
            self.eval()
            with torch.no_grad():
                dummy = torch.zeros(1, in_channels, input_len)
                out = self.conv1(dummy)
                out = self.conv2(out)
                return int(out.numel())
        finally:
            self.train(was_training)

    def forward(self, seq):
        # Forward through convolutional layers
        out = self.conv1(seq)
        if self.use_residual:
            out_res = out
        out = self.conv2(out)
        
        # Residual connection if enabled
        if self.use_residual:
            out_res = self.residual_pool(out_res)
            out = out + out_res
        
        # Flatten convolutional output before feeding into classifier
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

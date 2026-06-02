from .classifier import Classifier
import torch
import torch.nn as nn

class DeepCircCode(nn.Module):
    def __init__(
        self, 
        input_length=400, 
        num_conv_layers=3, 
        in_channels=4, 
        out_channels=[32, 64, 128], 
        kernel_size=5, 
        pool_size=2, 
    ):
        super(DeepCircCode, self).__init__()
        
        # Convolutional + Pooling Layers
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()  # Batch normalization layers

        current_in_channels = in_channels
        for i in range(num_conv_layers):
            self.conv_layers.append(
                nn.Conv1d(in_channels=current_in_channels, 
                          out_channels=out_channels[i], 
                          kernel_size=kernel_size, 
                          padding=kernel_size // 2)
            )
            self.bn_layers.append(nn.BatchNorm1d(out_channels[i]))  # BatchNorm after each conv layer
            self.pool_layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))
            current_in_channels = out_channels[i]

        # Calculate the size of the feature map after the convolution and pooling layers
        conv_output_size = input_length
        for _ in range(num_conv_layers):
            conv_output_size = (conv_output_size + 2 * (kernel_size // 2) - (kernel_size - 1) - 1) // 1 + 1
            conv_output_size = (conv_output_size - pool_size) // pool_size + 1

        # Fully Connected Layers
        self.classifier = Classifier(d_in=out_channels[-1] * conv_output_size)

    def forward(self, x):
        for i in range(len(self.conv_layers)):
            x = self.pool_layers[i](self.bn_layers[i](torch.relu(self.conv_layers[i](x))))

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x

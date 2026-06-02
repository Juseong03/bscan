import torch
import torch.nn as nn
import torch.nn.functional as F
from .classifier import Classifier

# Encoder-Decoder Autoencoder with hyperparameters
class Autoencoder(nn.Module):
    def __init__(self, input_channels, latent_dim, kernel_size1, kernel_size2, pool_size):
        super(Autoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=latent_dim, kernel_size=kernel_size1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Conv1d(in_channels=latent_dim, out_channels=latent_dim, kernel_size=kernel_size2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size)
        )
        
        # Decoder layers (reversing encoder)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=latent_dim, out_channels=latent_dim, kernel_size=kernel_size2, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=pool_size),
            nn.ConvTranspose1d(in_channels=latent_dim, out_channels=input_channels, kernel_size=kernel_size1, stride=1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=pool_size)
        )

    def forward(self, x):
        latent_space = self.encoder(x)
        reconstructed = self.decoder(latent_space)
        return latent_space, reconstructed


# Classifier using CNN on top of encoded latent space with hyperparameters
class CircNet(nn.Module):
    def __init__(
            self, 
            length_seq=400,
            input_channels=4, 
            latent_dim=128, 
            conv_filters=[64, 32], 
            kernel_size1=12, 
            kernel_size2=6, 
            pool_size=2
        ):
        super(CircNet, self).__init__()
        
        # Load the autoencoder with custom hyperparameters
        self.autoencoder = Autoencoder(input_channels=input_channels, latent_dim=latent_dim, 
                                       kernel_size1=kernel_size1, kernel_size2=kernel_size2, pool_size=pool_size)
        
        # CNN layers for classification
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=latent_dim, out_channels=conv_filters[0], kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Conv1d(in_channels=conv_filters[0], out_channels=conv_filters[1], kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size)
        )
        
        # Dynamically calculate the output size after the convolutional and pooling layers
        output_size = self._get_conv_output_size(length_seq, latent_dim, kernel_size1, kernel_size2, pool_size)
        # Fully connected layers for final classification
        self.flatten = nn.Flatten()
        self.classifier = Classifier(d_in=conv_filters[-1] * output_size)

    def forward(self, x):
        
        latent_space, _ = self.autoencoder(x)
        
        conv_output = self.conv_layers(latent_space)

        out = self.flatten(conv_output)

        out = self.classifier(out)
        
        return out

    def _get_conv_output_size(self, input_length, latent_dim, kernel_size1, kernel_size2, pool_size):
        """
        Helper function to calculate the output size of the CNN layers.
        """
        # Calculate size after first convolution and pooling
        conv1_output_size = (input_length - kernel_size1) // 1 + 1
        pool1_output_size = conv1_output_size // pool_size
        
        # Calculate size after second convolution and pooling
        conv2_output_size = (pool1_output_size - kernel_size2) // 1 + 1
        pool2_output_size = conv2_output_size // pool_size
        
        # Calculate size after third convolution and pooling
        conv3_output_size = (pool2_output_size - 3) // 1 + 1
        pool3_output_size = conv3_output_size // pool_size

        # Additional pooling to reduce size from 46 to 23, and padding to get 22
        final_output_size = (pool3_output_size // 2) - 1  # Padding adjustment
        
        return final_output_size

import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(
        self, 
        d_in, 
        d_hiddens=[128], 
        n_hidden_layers=1, 
        dropout=0.3, 
        activation=nn.GELU, 
        n_classes=2, 
        use_layernorm=False
    ):
        """
        Improved Classifier with multiple hidden layers, customizable activation functions, and optional LayerNorm.
        
        Args:
        - d_in: Input dimension.
        - d_hiddens: List of dimensions for the hidden layers.
        - n_hidden_layers: Number of hidden layers.
        - dropout: Dropout rate for regularization.
        - activation: Activation function to use between layers (default: GELU).
        - n_classes: Number of output classes.
        - use_layernorm: Boolean flag to apply LayerNorm after each hidden layer (default: False).
        """
        super(Classifier, self).__init__()

        layers = [nn.LayerNorm(d_in)]  # First batch normalization layer

        # Build hidden layers
        d_hidden_dims = d_hiddens * n_hidden_layers  # Repeat hidden dimensions if necessary
        prev_dim = d_in  # Track previous layer's output dimension

        for d_hidden in d_hidden_dims:
            layers.append(nn.Linear(prev_dim, d_hidden))  # Linear layer
            if use_layernorm:
                layers.append(nn.LayerNorm(d_hidden))  # Optional LayerNorm
            layers.append(activation())  # Non-linearity (GELU by default)
            layers.append(nn.Dropout(dropout))  # Dropout layer
            prev_dim = d_hidden  # Update previous dimension to the current hidden layer

        # Output layer
        layers.append(nn.Linear(prev_dim, n_classes))  # Final output layer

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

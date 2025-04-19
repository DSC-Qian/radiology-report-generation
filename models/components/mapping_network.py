import torch
import torch.nn as nn
import math


class MLPMapper(nn.Module):
    """
    Multi-layer perceptron to map image features to language model embeddings.
    
    Args:
        input_dim (int): Dimension of the input image features.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int): Dimension of the output embeddings.
        num_layers (int): Number of layers in the MLP.
        dropout (float): Dropout probability.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super(MLPMapper, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Create MLP layers
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            # Add activation and dropout for all but the last layer
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        return self.mlp(x)


class TransformerMapper(nn.Module):
    """
    Transformer-based mapper to convert image features to a sequence of embeddings.
    
    Args:
        input_dim (int): Dimension of the input image features.
        hidden_dim (int): Dimension of the transformer hidden layer.
        output_dim (int): Dimension of the output embeddings.
        num_layers (int): Number of transformer layers.
        num_heads (int): Number of attention heads.
        seq_len (int): Length of the output sequence.
        dropout (float): Dropout probability.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, 
                 num_heads=8, seq_len=16, dropout=0.1):
        super(TransformerMapper, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # Project input features to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Create learnable positional embeddings
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, seq_len, hidden_dim)
        )
        
        # Create transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Final projection to output dimension
        self.output_projection = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_dim).
        """
        batch_size = x.size(0)
        
        # Project input features to hidden dimension
        x = self.input_projection(x)
        
        # Expand to a sequence by repeating
        x = x.unsqueeze(1).expand(-1, self.seq_len, -1)
        
        # Add positional embeddings
        x = x + self.positional_embedding
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Project to output dimension
        x = self.output_projection(x)
        
        return x


def get_mapping_network(mapper_type='mlp', input_dim=2048, hidden_dim=768, output_dim=768,
                       num_layers=2, num_heads=8, seq_len=16, dropout=0.1):
    """
    Factory function to get a mapping network.
    
    Args:
        mapper_type (str): Type of the mapper ('mlp' or 'transformer').
        input_dim (int): Dimension of the input image features.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int): Dimension of the output embeddings.
        num_layers (int): Number of layers.
        num_heads (int): Number of attention heads (for transformer).
        seq_len (int): Length of the output sequence (for transformer).
        dropout (float): Dropout probability.
        
    Returns:
        nn.Module: Mapping network model.
    """
    if mapper_type == 'mlp':
        return MLPMapper(input_dim, hidden_dim, output_dim, num_layers, dropout)
    elif mapper_type == 'transformer':
        return TransformerMapper(input_dim, hidden_dim, output_dim, num_layers, 
                                num_heads, seq_len, dropout)
    else:
        raise ValueError(f"Unsupported mapper type: {mapper_type}") 
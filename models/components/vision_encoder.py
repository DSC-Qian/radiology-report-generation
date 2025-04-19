import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel, ViTConfig


class ResNetEncoder(nn.Module):
    """
    ResNet-based encoder for chest X-ray images.
    Uses a pre-trained ResNet model and optionally fine-tunes it.
    
    Args:
        model_name (str): Name of the ResNet model.
        pretrained (bool): Whether to use a pre-trained model.
        freeze (bool): Whether to freeze the model weights.
    """
    def __init__(self, model_name='resnet50', pretrained=True, freeze=True):
        super(ResNetEncoder, self).__init__()
        
        # Load pre-trained model
        if model_name == 'resnet18':
            self.model = models.resnet18(weights='DEFAULT' if pretrained else None)
            self.output_dim = 512
        elif model_name == 'resnet34':
            self.model = models.resnet34(weights='DEFAULT' if pretrained else None)
            self.output_dim = 512
        elif model_name == 'resnet50':
            self.model = models.resnet50(weights='DEFAULT' if pretrained else None)
            self.output_dim = 2048
        elif model_name == 'resnet101':
            self.model = models.resnet101(weights='DEFAULT' if pretrained else None)
            self.output_dim = 2048
        elif model_name == 'resnet152':
            self.model = models.resnet152(weights='DEFAULT' if pretrained else None)
            self.output_dim = 2048
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        # Remove the final fully connected layer
        self.model = nn.Sequential(*(list(self.model.children())[:-1]))
        
        # Freeze the model if required
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        x = self.model(x)
        return x.view(x.size(0), -1)  # Flatten the output


class ViTEncoder(nn.Module):
    """
    Vision Transformer (ViT) encoder for chest X-ray images.
    Uses a pre-trained ViT model and optionally fine-tunes it.
    
    Args:
        model_name (str): Name of the ViT model.
        pretrained (bool): Whether to use a pre-trained model.
        freeze (bool): Whether to freeze the model weights.
    """
    def __init__(self, model_name='google/vit-base-patch16-224', pretrained=True, freeze=True):
        super(ViTEncoder, self).__init__()
        
        # Load pre-trained ViT model
        if pretrained:
            self.model = ViTModel.from_pretrained(model_name)
        else:
            config = ViTConfig.from_pretrained(model_name)
            self.model = ViTModel(config)
        
        self.output_dim = self.model.config.hidden_size
        
        # Freeze the model if required
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor containing the [CLS] token representation 
                         of shape (batch_size, output_dim).
        """
        outputs = self.model(x)
        cls_token = outputs.last_hidden_state[:, 0, :]  # Get the [CLS] token representation
        return cls_token


def get_vision_encoder(encoder_type='resnet', model_name=None, pretrained=True, freeze=True):
    """
    Factory function to get a vision encoder.
    
    Args:
        encoder_type (str): Type of the encoder ('resnet' or 'vit').
        model_name (str): Name of the specific model.
        pretrained (bool): Whether to use a pre-trained model.
        freeze (bool): Whether to freeze the model weights.
        
    Returns:
        nn.Module: Vision encoder model.
    """
    if encoder_type == 'resnet':
        model_name = model_name or 'resnet50'
        return ResNetEncoder(model_name, pretrained, freeze)
    elif encoder_type == 'vit':
        model_name = model_name or 'google/vit-base-patch16-224'
        return ViTEncoder(model_name, pretrained, freeze)
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}") 
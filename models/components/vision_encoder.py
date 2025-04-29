import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig, ViTForImageClassification
from typing import Optional, Tuple, List, Union, Dict


class EnhancedViTEncoder(nn.Module):
    """
    Enhanced Vision Transformer (ViT) encoder for chest X-ray images.
    Provides flexible options for feature extraction to better capture details
    important for medical report generation.

    Args:
        model_name (str): Name of the ViT model from Hugging Face.
        pretrained (bool): Whether to use a pre-trained model.
        freeze_base (bool): Whether to freeze the base model weights.
        freeze_pattern (Optional[str]): Pattern for selective freezing ('none', 'embeddings_only', 'partial').
        unfreeze_layers (Optional[List[int]]): Specific layer indices to unfreeze (when freeze_pattern='partial').
        output_attentions (bool): Whether to output attention weights.
        output_hidden_states (bool): Whether to output hidden states from all layers.
        feature_selection (str): Method to select features ('cls', 'all', 'mean', 'attention_weighted').
        use_cls_token (bool): Whether to use the CLS token in the output.
        resolution (Optional[int]): Input resolution (will interpolate if model's native resolution differs).
    """
    def __init__(
        self,
        model_name: str = 'google/vit-base-patch16-224',
        pretrained: bool = True,
        freeze_base: bool = True,
        freeze_pattern: Optional[str] = 'none',
        unfreeze_layers: Optional[List[int]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        feature_selection: str = 'attention_weighted',
        use_cls_token: bool = True,
        resolution: Optional[int] = None
    ):
        super(EnhancedViTEncoder, self).__init__()

        # Load pre-trained ViT model with specified options
        if pretrained:
            self.model = ViTModel.from_pretrained(
                model_name,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                attn_implementation="eager"
            )
        else:
            config = ViTConfig.from_pretrained(
                model_name,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states
            )
            self.model = ViTModel(config)

        self.output_dim = self.model.config.hidden_size
        self.feature_selection = feature_selection
        self.use_cls_token = use_cls_token
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        
        # Create parameter to track original resolution
        self.original_resolution = self.model.config.image_size
        self.target_resolution = resolution or self.original_resolution
        
        # Implement smart freezing strategy
        self._apply_freezing_strategy(freeze_base, freeze_pattern, unfreeze_layers)
        
        # Add attention-based feature weighting if needed
        if feature_selection == 'attention_weighted':
            self.attention_weights = nn.Parameter(
                torch.ones(1, 1, 1) / self.model.config.hidden_size, 
                requires_grad=True
            )

    def _apply_freezing_strategy(self, freeze_base: bool, freeze_pattern: str, unfreeze_layers: Optional[List[int]] = None):
        """Apply strategic freezing to different parts of the model."""
        if freeze_base:
            # Freeze all parameters by default
            for param in self.model.parameters():
                param.requires_grad = False
                
            # Apply specific freezing pattern if requested
            if freeze_pattern == 'none':
                pass  # Keep everything frozen
            elif freeze_pattern == 'embeddings_only':
                # Unfreeze everything except embedding layers
                for name, param in self.model.named_parameters():
                    if not name.startswith('embeddings'):
                        param.requires_grad = True
            elif freeze_pattern == 'partial' and unfreeze_layers:
                # Unfreeze specific transformer layers
                for layer_idx in unfreeze_layers:
                    if layer_idx < len(self.model.encoder.layer):
                        for param in self.model.encoder.layer[layer_idx].parameters():
                            param.requires_grad = True

    def _extract_features(self, outputs):
        """Extract features based on the selected method."""
        last_hidden_state = outputs.last_hidden_state
        
        if self.feature_selection == 'cls':
            # Return only the CLS token representations
            return last_hidden_state[:, 0]
        
        elif self.feature_selection == 'all':
            # Return all token representations (including CLS if use_cls_token=True)
            if not self.use_cls_token:
                return last_hidden_state[:, 1:]
            return last_hidden_state
        
        elif self.feature_selection == 'mean':
            # Return mean of all token representations
            if self.use_cls_token:
                return torch.mean(last_hidden_state, dim=1)
            return torch.mean(last_hidden_state[:, 1:], dim=1)
        
        elif self.feature_selection == 'attention_weighted':
            # Weight features using learned attention weights
            if self.use_cls_token:
                features = last_hidden_state
            else:
                features = last_hidden_state[:, 1:]
                
            # Apply attention weighting
            attn_weights = torch.softmax(self.attention_weights, dim=-1)
            weighted_features = features * attn_weights
            return weighted_features
            
        else:
            raise ValueError(f"Unsupported feature selection method: {self.feature_selection}")

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the enhanced ViT encoder.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).

        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]: 
                - Default: Tensor with selected features
                - With output options: Dict containing features and requested outputs (attentions/hidden_states)
        """
        # Handle resolution mismatch if needed
        if self.original_resolution != self.target_resolution:
            x = nn.functional.interpolate(
                x, 
                size=(self.target_resolution, self.target_resolution),
                mode='bilinear',
                align_corners=False
            )
            
        # Forward pass through ViT
        outputs = self.model(pixel_values=x)
        
        # Extract features based on configured method
        features = self._extract_features(outputs)
        
        # Return additional outputs if requested
        if self.output_attentions or self.output_hidden_states:
            result = {"features": features}
            if self.output_attentions:
                result["attentions"] = outputs.attentions
            if self.output_hidden_states:
                result["hidden_states"] = outputs.hidden_states
            return result
            
        return features


def get_vision_encoder(
    model_name: str = 'google/vit-base-patch16-224',
    pretrained: bool = True,
    freeze_base: bool = True,
    freeze_pattern: str = 'none',
    unfreeze_layers: Optional[List[int]] = None,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
    feature_selection: str = 'cls',
    use_cls_token: bool = True,
    resolution: Optional[int] = None
) -> EnhancedViTEncoder:
    """
    Factory function to get an enhanced vision transformer encoder.
    
    Args:
        model_name (str): Name of the ViT model from Hugging Face.
        pretrained (bool): Whether to use a pre-trained model.
        freeze_base (bool): Whether to freeze the base model weights.
        freeze_pattern (str): Pattern for selective freezing ('none', 'embeddings_only', 'partial').
        unfreeze_layers (Optional[List[int]]): Specific layer indices to unfreeze (when freeze_pattern='partial').
        output_attentions (bool): Whether to output attention weights.
        output_hidden_states (bool): Whether to output hidden states from all layers.
        feature_selection (str): Method to select features ('cls', 'all', 'mean', 'attention_weighted').
        use_cls_token (bool): Whether to use the CLS token in the output.
        resolution (Optional[int]): Input resolution (will interpolate if model's native resolution differs).
        
    Returns:
        EnhancedViTEncoder: Enhanced vision transformer encoder.
    """
    return EnhancedViTEncoder(
        model_name=model_name,
        pretrained=pretrained,
        freeze_base=freeze_base,
        freeze_pattern=freeze_pattern,
        unfreeze_layers=unfreeze_layers,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        feature_selection=feature_selection,
        use_cls_token=use_cls_token,
        resolution=resolution
    ) 
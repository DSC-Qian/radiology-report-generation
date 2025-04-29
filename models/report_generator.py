import torch
import torch.nn as nn
from models.components.vision_encoder import get_vision_encoder
from models.components.language_decoder import get_language_decoder
from transformers import AutoTokenizer
from typing import Dict, List, Union, Optional, Tuple


class RadiologyReportGenerator(nn.Module):
    """
    End-to-end model for generating radiology reports from chest X-ray images.
    Uses Vision Transformer features with cross-attention in the GPT-2 decoder.

    Args:
        vision_model_name (str): Name of the ViT model from Hugging Face.
        gpt2_model_name (str): Specific GPT-2 model variant to use.
        freeze_vision (bool): Whether to freeze the vision encoder.
        freeze_decoder (bool): Whether to freeze the language decoder.
        pretrained_vision (bool): Whether to use a pre-trained vision encoder.
        pretrained_decoder (bool): Whether to use a pre-trained decoder.
        decoder_hidden_dim (int): Hidden dimension required by the decoder.
        tokenizer_name (str): Name of the tokenizer to use.
        max_length (int): Maximum length of the generated sequence.
        vision_feature_selection (str): Method to select vision features ('cls', 'all', 'mean', 'attention_weighted').
        vision_freeze_pattern (str): Pattern for selectively freezing vision encoder layers.
        vision_unfreeze_layers (List[int]): Specific vision encoder layers to unfreeze.
        output_attentions (bool): Whether to output attention weights.
        output_hidden_states (bool): Whether to output hidden states from all layers.
    """
    def __init__(
        self, 
        vision_model_name: str = 'google/vit-base-patch16-224',
        gpt2_model_name: str = 'gpt2', 
        freeze_vision: bool = True,
        freeze_decoder: bool = True, 
        pretrained_vision: bool = True, 
        pretrained_decoder: bool = True,
        decoder_hidden_dim: int = 768,
        tokenizer_name: str = 'gpt2', 
        max_length: int = 512,
        vision_feature_selection: str = 'all',
        vision_freeze_pattern: str = 'none',
        vision_unfreeze_layers: Optional[List[int]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        super().__init__()
        
        # Save parameters
        self.vision_model_name = vision_model_name
        self.gpt2_model_name = gpt2_model_name
        self.max_length = max_length
        self.decoder_hidden_dim = decoder_hidden_dim
        self.vision_feature_selection = vision_feature_selection
        
        # Save freeze parameters
        self.freeze_vision = freeze_vision
        self.freeze_decoder = freeze_decoder
        
        # Init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side='left')
        
        # Configure tokenizer properly for GPT2
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # Update the tokenizer's vocabulary with the pad token
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        
        # Set padding side to left for autoregressive models like GPT2
        self.tokenizer.padding_side = 'left'
        
        # Define enhanced vision encoder with the selected feature extraction method
        self.vision_encoder = get_vision_encoder(
            model_name=vision_model_name,
            pretrained=pretrained_vision,
            freeze_base=freeze_vision,
            freeze_pattern=vision_freeze_pattern,
            unfreeze_layers=vision_unfreeze_layers,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            feature_selection=vision_feature_selection,
            use_cls_token=True
        )
        
        # Get the output dimension from the vision encoder
        encoder_dim = self.vision_encoder.output_dim
        
        # --- Define projection layer ---
        # This projection setup depends on the feature_selection method
        if vision_feature_selection in ['cls', 'mean']:
            # For CLS token or mean pooling, we get a single vector per image
            # We need to project and expand to create a sequence
            self.use_single_vector = True
            self.vision_projection = nn.Linear(encoder_dim, self.decoder_hidden_dim)
            # Generate a sequence of tokens from the single vector
            self.seq_len = 16  # Number of tokens to generate
            self.create_sequence = True
            
            # Positional embeddings for expanding to a sequence
            self.position_embeddings = nn.Parameter(
                torch.zeros(1, self.seq_len, self.decoder_hidden_dim)
            )
        else:
            # For 'all' or 'attention_weighted', we already have a sequence
            self.use_single_vector = False
            self.vision_projection = nn.Linear(encoder_dim, self.decoder_hidden_dim)
            self.create_sequence = False
        # --- End projection layer ---
        
        # Define GPT-2 language decoder with cross-attention
        self.language_decoder = get_language_decoder(
            model_name=gpt2_model_name,
            pretrained=pretrained_decoder,
            freeze=freeze_decoder
        )
        
        # Ensure the decoder's underlying model config uses the correct pad token ID
        self.language_decoder.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Check if decoder's hidden size matches the projection target
        decoder_hidden_size = getattr(self.language_decoder.model.config, 'hidden_size', 
                                     getattr(self.language_decoder.model.config, 'n_embd', None))
        
        if decoder_hidden_size and decoder_hidden_size != self.decoder_hidden_dim:
             print(f"Warning: Decoder hidden size ({decoder_hidden_size}) "
                   f"differs from target projection dimension ({self.decoder_hidden_dim}). "
                   f"Ensure `decoder_hidden_dim` matches the decoder's requirements.")
        
        # Initialize weights
        self._init_weights()
        
        # --- Start: Training configuration reminders ---
        if freeze_decoder:
             print("Warning: freeze_decoder is set to True. The language decoder weights will not be trained. "
                   "This should typically be False during active training stages.")
        # --- End: Training configuration reminders ---
    
    def _init_weights(self):
        """
        Initialize weights for the model.
        """
        # Initialize weights for the projection layer
        nn.init.xavier_uniform_(self.vision_projection.weight)
        if self.vision_projection.bias is not None:
            nn.init.zeros_(self.vision_projection.bias)
            
        # Initialize position embeddings if used
        if hasattr(self, 'position_embeddings'):
            nn.init.normal_(self.position_embeddings, mean=0.0, std=0.02)
    
    def _process_vision_features(self, features):
        """
        Process vision features based on the selected feature extraction method.
        
        Args:
            features: Output from the vision encoder, could be tensor or dict
            
        Returns:
            projected_features: Features projected to decoder's dimension
            encoder_attention_mask: Attention mask for the encoder features
        """
        # Handle both tensor and dictionary outputs from the enhanced encoder
        if isinstance(features, dict):
            image_features = features["features"]
        else:
            image_features = features
            
        batch_size = image_features.size(0)
        
        # Handle features based on their dimensionality
        if self.use_single_vector:
            # Case: Single vector per image (cls token or mean pooling)
            if len(image_features.shape) == 2:  # [B, D]
                # Project to decoder dimension
                projected_vector = self.vision_projection(image_features)
                
                if self.create_sequence:
                    # Expand to sequence and add positional embeddings
                    projected_features = projected_vector.unsqueeze(1).expand(-1, self.seq_len, -1)
                    projected_features = projected_features + self.position_embeddings
                    # Create attention mask (all ones)
                    encoder_attention_mask = torch.ones(batch_size, self.seq_len, 
                                                      device=projected_features.device, dtype=torch.long)
                else:
                    # Keep as a single token
                    projected_features = projected_vector.unsqueeze(1)  # [B, 1, D]
                    encoder_attention_mask = torch.ones(batch_size, 1, 
                                                      device=projected_features.device, dtype=torch.long)
            else:
                raise ValueError(f"Expected 2D tensor for 'cls' or 'mean' feature selection, got shape {image_features.shape}")
        else:
            # Case: Sequence of vectors (all tokens or attention weighted)
            if len(image_features.shape) == 3:  # [B, S, D]
                # Project each token to decoder dimension
                projected_features = self.vision_projection(image_features)
                # Create attention mask (all ones)
                encoder_attention_mask = torch.ones(batch_size, projected_features.size(1), 
                                                   device=projected_features.device, dtype=torch.long)
            else:
                raise ValueError(f"Expected 3D tensor for 'all' or 'attention_weighted' feature selection, got shape {image_features.shape}")
                
        return projected_features, encoder_attention_mask
    
    def forward(self, images, input_ids=None, attention_mask=None, labels=None):
        """
        Forward pass through the model.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, channels, height, width).
            input_ids (torch.Tensor, optional): Input token IDs for the decoder.
            attention_mask (torch.Tensor, optional): Attention mask for the decoder.
            labels (torch.Tensor, optional): Labels for language modeling.
            
        Returns:
            dict: Model outputs.
        """
        # Set appropriate train/eval mode
        if not self.freeze_vision:
            self.vision_encoder.train()
        else:
            self.vision_encoder.eval()
        
        # Extract image features using the vision encoder
        with torch.set_grad_enabled(not self.freeze_vision):
            vision_outputs = self.vision_encoder(images)
        
        # Process and project vision features
        encoder_hidden_states, encoder_attention_mask = self._process_vision_features(vision_outputs)
        
        # Pass additional vision outputs if available
        additional_encoder_outputs = {}
        if isinstance(vision_outputs, dict):
            if 'attentions' in vision_outputs:
                additional_encoder_outputs['encoder_attentions'] = vision_outputs['attentions']
            if 'hidden_states' in vision_outputs:
                additional_encoder_outputs['encoder_hidden_states'] = vision_outputs['hidden_states']
        
        # Prepare inputs for the decoder
        if input_ids is not None and labels is not None:
            # Training/Validation with Labels: Use provided input_ids and cross-attention
            outputs = self.language_decoder(
                inputs=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=True,
                **additional_encoder_outputs
            )
            
            return {
                "loss": outputs.loss,
                "logits": outputs.logits
            }
        else:
            # Inference Mode: Generate text using encoder_hidden_states
            # Use a structured textual prefix for generation
            prefix_text = "FINAL REPORT  "
            # Tokenize the prefix and move to the same device
            prefix_ids = self.tokenizer(prefix_text, return_tensors="pt").input_ids.to(encoder_hidden_states.device)
            # Expand the prefix tokens to match the batch size
            batch_size = images.size(0)
            prompt_ids = prefix_ids.expand(batch_size, -1)
            prompt_attention_mask = torch.ones_like(prompt_ids)

            # Generate using the decoder's generate method with cross-attention
            outputs = self.language_decoder.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                max_length=self.max_length,
                min_length=10,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            return {"generated_ids": outputs}
    
    def generate(self, images):
        """
        Generate reports from images using the forward pass in inference mode.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, channels, height, width).
            
        Returns:
            list: List of generated report texts.
        """
        # Set model to evaluation mode
        self.eval()
        
        # Perform inference using the forward pass
        with torch.no_grad():
            outputs = self.forward(images=images)
        
        generated_ids = outputs["generated_ids"]
        
        # Decode generated tokens to text
        generated_texts = []
        for ids in generated_ids:
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            generated_texts.append(text.strip())
        
        return generated_texts


def get_report_generator(config):
    """
    Factory function to get a complete report generator model.
    
    Args:
        config (dict): Configuration dictionary with model parameters.
        
    Returns:
        RadiologyReportGenerator: Complete model.
    """
    return RadiologyReportGenerator(
        vision_model_name=config.get('vision_model_name', 'google/vit-base-patch16-224'),
        gpt2_model_name=config.get('gpt2_model_name', 'gpt2'),
        freeze_vision=config.get('freeze_vision', True),
        freeze_decoder=config.get('freeze_decoder', True),
        pretrained_vision=config.get('pretrained_vision', True),
        pretrained_decoder=config.get('pretrained_decoder', True),
        decoder_hidden_dim=config.get('decoder_hidden_dim', 768),
        tokenizer_name=config.get('tokenizer_name', 'gpt2'),
        max_length=config.get('max_length', 512),
        vision_feature_selection=config.get('vision_feature_selection', 'all'),
        vision_freeze_pattern=config.get('vision_freeze_pattern', 'none'),
        vision_unfreeze_layers=config.get('vision_unfreeze_layers', None),
        output_attentions=config.get('output_attentions', False),
        output_hidden_states=config.get('output_hidden_states', False)
    ) 
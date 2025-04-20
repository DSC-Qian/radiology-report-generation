import torch
import torch.nn as nn
from models.components.vision_encoder import get_vision_encoder
from models.components.mapping_network import get_mapping_network
from models.components.language_decoder import get_language_decoder
from transformers import AutoTokenizer


class RadiologyReportGenerator(nn.Module):
    """
    End-to-end model for generating radiology reports from chest X-ray images.
    
    Args:
        encoder_type (str): Type of vision encoder ('resnet' or 'vit').
        encoder_model (str): Specific encoder model name.
        mapper_type (str): Type of mapping network ('mlp' or 'transformer').
        decoder_type (str): Type of language decoder ('gpt2' or 'biomedical').
        decoder_model (str): Specific decoder model name.
        freeze_encoder (bool): Whether to freeze the vision encoder.
        freeze_decoder (bool): Whether to freeze the language decoder.
        pretrained_encoder (bool): Whether to use a pre-trained encoder.
        pretrained_decoder (bool): Whether to use a pre-trained decoder.
        mapper_hidden_dim (int): Hidden dimension of the mapping network.
        mapper_num_layers (int): Number of layers in the mapping network.
        mapper_num_heads (int): Number of attention heads in the transformer mapper.
        mapper_seq_len (int): Length of the sequence in the transformer mapper.
        mapper_dropout (float): Dropout probability in the mapping network.
        tokenizer_name (str): Name of the tokenizer to use.
        max_length (int): Maximum length of the generated sequence.
    """
    def __init__(self, encoder_type='resnet', encoder_model='resnet50',
                 mapper_type='transformer', mapper_hidden_dim=768, mapper_num_layers=2,
                 mapper_num_heads=8, mapper_seq_len=16, mapper_dropout=0.1,
                 decoder_type='gpt2', decoder_model='gpt2', freeze_encoder=True,
                 freeze_decoder=True, pretrained_encoder=True, pretrained_decoder=True,
                 tokenizer_name='gpt2', max_length=512):
        super().__init__()
        
        # Save parameters
        self.encoder_type = encoder_type
        self.encoder_model = encoder_model
        self.mapper_type = mapper_type
        self.decoder_type = decoder_type
        self.decoder_model = decoder_model
        self.max_length = max_length
        
        # Save freeze parameters
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        
        # Init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Configure tokenizer properly for GPT2 and other models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # Update the tokenizer's vocabulary with the pad token
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        
        # Set padding side to left for autoregressive models like GPT2
        self.tokenizer.padding_side = 'left'
        
        # Define encoder
        self.vision_encoder = get_vision_encoder(
            encoder_type,
            encoder_model,
            pretrained=pretrained_encoder,
            freeze=freeze_encoder
        )
        
        # Define encoder output dimension
        if encoder_type == 'resnet':
            if encoder_model == 'resnet18' or encoder_model == 'resnet34':
                encoder_dim = 512
            else:  # resnet50, resnet101, resnet152
                encoder_dim = 2048
        else:  # ViT
            encoder_dim = 768  # For base ViT models
        
        # Define mapping network
        self.mapping_network = get_mapping_network(
            mapper_type,
            input_dim=encoder_dim,
            hidden_dim=mapper_hidden_dim,
            output_dim=mapper_hidden_dim,
            num_layers=mapper_num_layers,
            num_heads=mapper_num_heads,
            seq_len=mapper_seq_len,
            dropout=mapper_dropout
        )
        
        # Define embedding adapter
        if mapper_type == 'transformer':
            # For transformer mapper, adapt the flattened output to decoder embedding dim
            mapper_output_dim = mapper_hidden_dim * mapper_seq_len
            self.embedding_adapter = nn.Linear(mapper_output_dim, mapper_hidden_dim)
        
        # Define language decoder
        self.language_decoder = get_language_decoder(
            decoder_type,
            decoder_model,
            pretrained=pretrained_decoder,
            freeze=freeze_decoder
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights for the model.
        """
        # Initialize weights for mapping network
        if hasattr(self.mapping_network, 'apply'):
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            
            self.mapping_network.apply(init_weights)
        
        # Initialize weights for embedding adapter if exists
        if hasattr(self, 'embedding_adapter'):
            nn.init.xavier_uniform_(self.embedding_adapter.weight)
            nn.init.zeros_(self.embedding_adapter.bias)
    
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
        # Make sure the model is in train mode for the active parts
        if not self.freeze_encoder:
            self.vision_encoder.train()
        else:
            self.vision_encoder.eval()
        
        # Extract image features using the vision encoder
        with torch.set_grad_enabled(not self.freeze_encoder):
            image_features = self.vision_encoder(images)
        
        # If image features don't require grad but should, enable it
        if not self.freeze_encoder and not image_features.requires_grad:
            image_features.requires_grad_(True)
        
        # Map image features to decoder-compatible embeddings
        if self.mapper_type == 'mlp':
            # MLP mapper outputs a single embedding
            mapped_features = self.mapping_network(image_features)
            
            # Add batch dimension if needed
            if len(mapped_features.shape) == 1:
                mapped_features = mapped_features.unsqueeze(0)
                
            # Prepare inputs for the decoder
            if input_ids is not None:
                # Training mode: Use provided input_ids
                outputs = self.language_decoder(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Ensure loss has gradients
                if not outputs.loss.requires_grad:
                    # This shouldn't normally happen, but let's handle it just in case
                    print("Warning: Loss doesn't require gradients. This may cause backpropagation issues.")
                
                return {
                    "loss": outputs.loss,
                    "logits": outputs.logits
                }
            else:
                # Inference mode: Generate from the model
                # Create a prompt with BOS token
                prompt_ids = torch.full(
                    (image_features.size(0), 1),
                    self.tokenizer.bos_token_id,
                    dtype=torch.long,
                    device=image_features.device
                )
                
                outputs = self.language_decoder.generate(
                    input_ids=prompt_ids,
                    max_length=self.max_length,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    min_length=10
                )
                
                return {"generated_ids": outputs}
                
        else:  # transformer mapper
            # Transformer mapper outputs a sequence of embeddings
            mapped_features = self.mapping_network(image_features)
            
            # Flatten the sequence for the embedding adapter
            batch_size = mapped_features.size(0)
            flat_features = mapped_features.view(batch_size, -1)
            
            # Adapt to decoder embeddings
            adapted_features = self.embedding_adapter(flat_features)
            
            # Prepare inputs for the decoder
            if input_ids is not None:
                # Training mode: Use provided input_ids
                outputs = self.language_decoder(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Ensure loss has gradients
                if not outputs.loss.requires_grad:
                    # This shouldn't normally happen, but let's handle it just in case
                    print("Warning: Loss doesn't require gradients. This may cause backpropagation issues.")
                
                return {
                    "loss": outputs.loss,
                    "logits": outputs.logits
                }
            else:
                # Inference mode: Generate from the model
                # Create a prompt with BOS token
                prompt_ids = torch.full(
                    (image_features.size(0), 1),
                    self.tokenizer.bos_token_id,
                    dtype=torch.long,
                    device=image_features.device
                )
                
                outputs = self.language_decoder.generate(
                    input_ids=prompt_ids,
                    max_length=self.max_length,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    min_length=10
                )
                
                return {"generated_ids": outputs}
    
    def generate(self, images):
        """
        Generate reports from images.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, channels, height, width).
            
        Returns:
            list: List of generated report texts.
        """
        # Set model to evaluation mode
        self.eval()
        
        # Extract image features
        with torch.no_grad():
            image_features = self.vision_encoder(images)
        
        # Map image features to decoder-compatible embeddings
        if self.mapper_type == 'mlp':
            mapped_features = self.mapping_network(image_features)
            
            # Add batch dimension if needed
            if len(mapped_features.shape) == 1:
                mapped_features = mapped_features.unsqueeze(0)
                
            # Create a prompt with BOS token
            prompt_ids = torch.full(
                (image_features.size(0), 1),
                self.tokenizer.bos_token_id,
                dtype=torch.long,
                device=image_features.device
            )
            
            # Create attention mask - all 1s for the prompt
            attention_mask = torch.ones_like(prompt_ids)
            
            # Generate text
            with torch.no_grad():
                outputs = self.language_decoder.generate(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    min_length=10
                )
        else:  # transformer mapper
            # Transformer mapper outputs a sequence of embeddings
            mapped_features = self.mapping_network(image_features)
            
            # Flatten the sequence for the embedding adapter
            batch_size = mapped_features.size(0)
            flat_features = mapped_features.view(batch_size, -1)
            
            # Adapt to decoder embeddings
            adapted_features = self.embedding_adapter(flat_features)
            
            # Create a prompt with BOS token
            prompt_ids = torch.full(
                (image_features.size(0), 1),
                self.tokenizer.bos_token_id,
                dtype=torch.long,
                device=image_features.device
            )
            
            # Create attention mask - all 1s for the prompt
            attention_mask = torch.ones_like(prompt_ids)
            
            # Generate text
            with torch.no_grad():
                outputs = self.language_decoder.generate(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    min_length=10
                )
        
        # Decode generated tokens to text
        generated_texts = []
        for ids in outputs:
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
        encoder_type=config.get('encoder_type', 'resnet'),
        encoder_model=config.get('encoder_model', 'resnet50'),
        mapper_type=config.get('mapper_type', 'transformer'),
        mapper_hidden_dim=config.get('mapper_hidden_dim', 768),
        mapper_num_layers=config.get('mapper_num_layers', 2),
        mapper_num_heads=config.get('mapper_num_heads', 8),
        mapper_seq_len=config.get('mapper_seq_len', 16),
        mapper_dropout=config.get('mapper_dropout', 0.1),
        decoder_type=config.get('decoder_type', 'gpt2'),
        decoder_model=config.get('decoder_model', 'gpt2'),
        freeze_encoder=config.get('freeze_encoder', True),
        freeze_decoder=config.get('freeze_decoder', True),
        pretrained_encoder=config.get('pretrained_encoder', True),
        pretrained_decoder=config.get('pretrained_decoder', True),
        tokenizer_name=config.get('tokenizer_name', 'gpt2'),
        max_length=config.get('max_length', 512)
    ) 
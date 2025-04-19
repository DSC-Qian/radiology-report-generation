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
        super(RadiologyReportGenerator, self).__init__()
        
        # Save configuration
        self.encoder_type = encoder_type
        self.mapper_type = mapper_type
        self.decoder_type = decoder_type
        self.max_length = max_length
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize vision encoder
        self.vision_encoder = get_vision_encoder(
            encoder_type=encoder_type,
            model_name=encoder_model,
            pretrained=pretrained_encoder,
            freeze=freeze_encoder
        )
        
        # Initialize language decoder
        self.language_decoder = get_language_decoder(
            decoder_type=decoder_type,
            model_name=decoder_model,
            pretrained=pretrained_decoder,
            freeze_encoder=freeze_decoder
        )
        
        # Initialize mapping network
        encoder_output_dim = self.vision_encoder.output_dim
        decoder_embedding_dim = self.language_decoder.embedding_dim
        
        self.mapping_network = get_mapping_network(
            mapper_type=mapper_type,
            input_dim=encoder_output_dim,
            hidden_dim=mapper_hidden_dim,
            output_dim=decoder_embedding_dim,
            num_layers=mapper_num_layers,
            num_heads=mapper_num_heads,
            seq_len=mapper_seq_len,
            dropout=mapper_dropout
        )
        
        # If using transformer mapper, need embedding adapter
        if mapper_type == 'transformer':
            # Create embeddings adapter to convert mapped features to decoder inputs
            self.embedding_adapter = nn.Linear(
                mapper_seq_len * decoder_embedding_dim,
                decoder_embedding_dim
            )
        
        # Store training configuration
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
    
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
                    early_stopping=True
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
                    early_stopping=True
                )
                
                return {"generated_ids": outputs}
    
    def generate(self, images):
        """
        Generate reports from images.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, channels, height, width).
            
        Returns:
            list: Generated report texts.
        """
        # Set model to evaluation mode
        self.eval()
        
        with torch.no_grad():
            # Forward pass to get generated IDs
            outputs = self.forward(images)
            generated_ids = outputs["generated_ids"]
            
            # Convert IDs to text
            generated_texts = []
            for ids in generated_ids:
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
                generated_texts.append(text)
        
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
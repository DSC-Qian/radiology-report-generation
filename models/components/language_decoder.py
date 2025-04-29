import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
from typing import Optional, Dict, Any, Union, Tuple


class GPT2Decoder(nn.Module):
    """
    GPT-2 based decoder for generating radiology reports.
    Uses cross-attention to attend to vision features from the encoder.
    
    Args:
        model_name (str): Name of the pre-trained GPT-2 model.
        pretrained (bool): Whether to use a pre-trained model.
        freeze_encoder (bool): Whether to freeze the transformer encoder.
        vocab_size (int): Size of the vocabulary (for non-pretrained models).
        embedding_dim (int): Dimension of the embeddings (for non-pretrained models).
    """
    def __init__(
        self, 
        model_name: str = 'gpt2', 
        pretrained: bool = True, 
        freeze_encoder: bool = True, 
        vocab_size: Optional[int] = None, 
        embedding_dim: int = 768
    ):
        super(GPT2Decoder, self).__init__()
        
        # Load pre-trained GPT-2 model with cross-attention enabled
        # Always load the base config, enable cross-attention, then load or init the model
        config = GPT2Config.from_pretrained(model_name)
        config.add_cross_attention = True  # This is crucial for attending to vision features
        config.return_dict = True
        
        if pretrained:
            self.model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
        else:
            if vocab_size is not None:
                config.vocab_size = vocab_size
            self.model = GPT2LMHeadModel(config)
        
        self.embedding_dim = self.model.config.hidden_size
        
        # Freeze the transformer encoder if required
        if freeze_encoder:
            for name, param in self.model.named_parameters():
                if 'lm_head' not in name:  # Only allow lm_head to be trained
                    param.requires_grad = False
    
    def forward(self, 
               inputs: torch.Tensor, 
               attention_mask: Optional[torch.Tensor] = None, 
               labels: Optional[torch.Tensor] = None, 
               past_key_values: Optional[Tuple[torch.Tensor]] = None,
               use_cache: Optional[bool] = None, 
               return_dict: Optional[bool] = None, 
               encoder_hidden_states: Optional[torch.Tensor] = None,
               encoder_attention_mask: Optional[torch.Tensor] = None,
               **kwargs) -> Dict[str, Any]:
        """
        Forward pass through the model.
        
        Args:
            inputs (torch.Tensor): Input IDs for the decoder.
            attention_mask (torch.Tensor): Attention mask.
            labels (torch.Tensor): Labels for language modeling.
            past_key_values: Past key values used for faster decoding.
            use_cache (bool): Whether to use past key values.
            return_dict (bool): Whether to return a dictionary or a tuple.
            encoder_hidden_states (torch.Tensor): Hidden states from the vision encoder.
            encoder_attention_mask (torch.Tensor): Attention mask for the encoder.
            
        Returns:
            Dict: Output of the GPT-2 model.
        """
        # Ensure labels are detached from the input_ids graph if they are the same tensor
        if labels is not None and torch.all(inputs == labels):
            labels = inputs.detach().clone()
        
        # Make sure to set the model to train or eval mode accordingly
        training_mode = labels is not None
        self.model.train(training_mode)
        
        outputs = self.model(
            input_ids=inputs,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,  # Always return dict for consistent handling
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            **kwargs
        )
        
        # Handle loss requires_grad issue
        if training_mode and not outputs.loss.requires_grad:
            # Create a wrapped loss that requires gradients
            wrapped_loss = outputs.loss + 0 * sum(p.sum() for p in self.model.parameters() if p.requires_grad)
            outputs.loss = wrapped_loss
        
        return outputs
    
    def generate(self, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None, 
                max_length: int = 512, 
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Generate text using the model with cross-attention to vision features.
        
        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, seq_len).
            max_length (int): Maximum length of the generated sequence.
            encoder_hidden_states (torch.Tensor): Hidden states from the vision encoder.
            encoder_attention_mask (torch.Tensor): Attention mask for the encoder.
            
        Returns:
            torch.Tensor: Generated tokens.
        """
        # Set to eval mode for generation
        self.model.eval()
        
        # Prepare encoder inputs for generation if provided
        generation_kwargs = {}
        if encoder_hidden_states is not None:
            generation_kwargs["encoder_hidden_states"] = encoder_hidden_states
            
            if encoder_attention_mask is not None:
                generation_kwargs["encoder_attention_mask"] = encoder_attention_mask
        
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            **{**generation_kwargs, **kwargs}
        )


def get_language_decoder(
    model_name: str = 'gpt2', 
    pretrained: bool = True, 
    freeze: bool = True, 
    vocab_size: Optional[int] = None, 
    embedding_dim: int = 768
) -> nn.Module:
    """
    Factory function to get a GPT-2 language decoder with cross-attention capabilities.
    
    Args:
        model_name (str): Name of the GPT-2 variant to use.
        pretrained (bool): Whether to use a pre-trained model.
        freeze (bool): Whether to freeze the model parameters.
        vocab_size (int): Size of the vocabulary (for non-pretrained models).
        embedding_dim (int): Dimension of the embeddings (for non-pretrained models).
        
    Returns:
        nn.Module: GPT-2 language decoder model.
    """
    return GPT2Decoder(
        model_name=model_name, 
        pretrained=pretrained, 
        freeze_encoder=freeze,
        vocab_size=vocab_size, 
        embedding_dim=embedding_dim
    ) 
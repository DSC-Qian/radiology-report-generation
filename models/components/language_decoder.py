import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, AutoConfig, AutoModelForCausalLM


class GPT2Decoder(nn.Module):
    """
    GPT-2 based decoder for generating radiology reports.
    
    Args:
        model_name (str): Name of the pre-trained GPT-2 model.
        pretrained (bool): Whether to use a pre-trained model.
        freeze_encoder (bool): Whether to freeze the transformer encoder.
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embeddings.
    """
    def __init__(self, model_name='gpt2', pretrained=True, freeze_encoder=True, 
                 vocab_size=None, embedding_dim=768):
        super(GPT2Decoder, self).__init__()
        
        # Load pre-trained GPT-2 model
        if pretrained:
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
        else:
            config = GPT2Config.from_pretrained(model_name)
            if vocab_size is not None:
                config.vocab_size = vocab_size
            self.model = GPT2LMHeadModel(config)
        
        self.embedding_dim = self.model.config.hidden_size
        
        # Freeze the transformer encoder if required
        if freeze_encoder:
            for param in self.model.transformer.parameters():
                param.requires_grad = False
            
            # Make sure lm_head is trainable even if transformer is frozen
            for param in self.model.lm_head.parameters():
                param.requires_grad = True
    
    def forward(self, inputs, attention_mask=None, labels=None, past_key_values=None,
               use_cache=None, return_dict=None):
        """
        Forward pass through the model.
        
        Args:
            inputs (torch.Tensor): Input IDs or embeddings, depending on input_type.
            attention_mask (torch.Tensor): Attention mask.
            labels (torch.Tensor): Labels for language modeling.
            past_key_values: Past key values used for faster decoding.
            use_cache (bool): Whether to use past key values.
            return_dict (bool): Whether to return a dictionary or a tuple.
            
        Returns:
            tuple or transformers.modeling_outputs.CausalLMOutputWithCrossAttentions:
                Output of the GPT-2 model.
        """
        # Ensure labels are detached from the input_ids graph if they are the same tensor
        # This prevents issues with double backpropagation
        if labels is not None and torch.all(inputs == labels):
            labels = inputs.detach().clone()
        
        # Make sure to set the model to train or eval mode accordingly
        original_mode = self.model.training
        if any(p.requires_grad for p in self.model.parameters()):
            self.model.train()
        
        outputs = self.model(
            input_ids=inputs,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True  # Always return dict for consistent handling
        )
        
        # Restore original mode
        self.model.train(original_mode)
        
        # Make sure the loss requires grad for backpropagation
        if labels is not None and not outputs.loss.requires_grad:
            print("Warning: GPT2 loss doesn't require gradients!")
        
        return outputs
    
    def generate(self, input_ids, attention_mask=None, max_length=512, **kwargs):
        """
        Generate text using the model.
        
        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, seq_len).
            max_length (int): Maximum length of the generated sequence.
            
        Returns:
            torch.Tensor: Generated tokens.
        """
        # Set to eval mode for generation
        self.model.eval()
        
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            **kwargs
        )


class BiomedicalDecoder(nn.Module):
    """
    Biomedical language model decoder for generating radiology reports.
    Uses models like BioGPT or Clinical models.
    
    Args:
        model_name (str): Name of the pre-trained biomedical language model.
        pretrained (bool): Whether to use a pre-trained model.
        freeze_encoder (bool): Whether to freeze the transformer encoder.
    """
    def __init__(self, model_name='microsoft/biogpt', pretrained=True, freeze_encoder=True):
        super(BiomedicalDecoder, self).__init__()
        
        # Load pre-trained biomedical language model
        if pretrained:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_config(config)
        
        self.embedding_dim = self.model.config.hidden_size
        
        # Freeze the transformer encoder if required
        if freeze_encoder:
            for name, param in self.model.named_parameters():
                if 'lm_head' not in name:  # Only allow lm_head to be trained
                    param.requires_grad = False
    
    def forward(self, inputs, attention_mask=None, labels=None, past_key_values=None,
               use_cache=None, return_dict=None):
        """
        Forward pass through the model.
        
        Args:
            inputs (torch.Tensor): Input IDs.
            attention_mask (torch.Tensor): Attention mask.
            labels (torch.Tensor): Labels for language modeling.
            past_key_values: Past key values used for faster decoding.
            use_cache (bool): Whether to use past key values.
            return_dict (bool): Whether to return a dictionary or a tuple.
            
        Returns:
            tuple or transformers.modeling_outputs.CausalLMOutputWithCrossAttentions:
                Output of the model.
        """
        # Ensure labels are detached from the input_ids graph if they are the same tensor
        if labels is not None and torch.all(inputs == labels):
            labels = inputs.detach().clone()
        
        # Make sure to set the model to train or eval mode accordingly
        original_mode = self.model.training
        if any(p.requires_grad for p in self.model.parameters()):
            self.model.train()
        
        outputs = self.model(
            input_ids=inputs,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True  # Always return dict for consistent handling
        )
        
        # Restore original mode
        self.model.train(original_mode)
        
        # Make sure the loss requires grad for backpropagation
        if labels is not None and not outputs.loss.requires_grad:
            print("Warning: Biomedical model loss doesn't require gradients!")
        
        return outputs
    
    def generate(self, input_ids, attention_mask=None, max_length=512, **kwargs):
        """
        Generate text using the model.
        
        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, seq_len).
            max_length (int): Maximum length of the generated sequence.
            
        Returns:
            torch.Tensor: Generated tokens.
        """
        # Set to eval mode for generation
        self.model.eval()
        
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            **kwargs
        )


def get_language_decoder(decoder_type='gpt2', model_name=None, pretrained=True, 
                        freeze_encoder=True, vocab_size=None, embedding_dim=768):
    """
    Factory function to get a language decoder.
    
    Args:
        decoder_type (str): Type of the decoder ('gpt2' or 'biomedical').
        model_name (str): Name of the specific model.
        pretrained (bool): Whether to use a pre-trained model.
        freeze_encoder (bool): Whether to freeze the transformer encoder.
        vocab_size (int): Size of the vocabulary (for non-pretrained models).
        embedding_dim (int): Dimension of the embeddings (for non-pretrained models).
        
    Returns:
        nn.Module: Language decoder model.
    """
    if decoder_type == 'gpt2':
        model_name = model_name or 'gpt2'
        return GPT2Decoder(model_name, pretrained, freeze_encoder, vocab_size, embedding_dim)
    elif decoder_type == 'biomedical':
        model_name = model_name or 'microsoft/biogpt'
        return BiomedicalDecoder(model_name, pretrained, freeze_encoder)
    else:
        raise ValueError(f"Unsupported decoder type: {decoder_type}") 
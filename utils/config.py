import os
import json
import argparse
from typing import Optional, List


class Config:
    """
    Configuration class for the Radiology Report Generation project.
    """
    def __init__(self, args=None):
        """
        Initialize configuration with default values.
        
        Args:
            args (argparse.Namespace, optional): Command-line arguments.
        """
        # Project paths
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(self.project_root, '../data/mimic-cxr-jpg/')
        self.csv_file = os.path.join(self.data_path, 'mimic-cxr-list-filtered.csv')
        
        # Create output directories if they don't exist
        self.output_dir = os.path.join(self.project_root, 'outputs')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        self.result_dir = os.path.join(self.output_dir, 'results')
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Model configuration
        self.model_config = {
            # Vision encoder settings
            'vision_model_name': 'google/vit-base-patch16-224',  # Base ViT model for feature extraction
            'pretrained_vision': True,  # Always use pretrained vision models
            'freeze_vision': True,  # Stage 1: True, Stage 2: False
            
            # Vision encoder feature extraction settings
            'vision_feature_selection': 'attention_weighted',  # 'cls', 'all', 'mean', 'attention_weighted'
            'vision_freeze_pattern': 'embeddings_only',  # 'none', 'embeddings_only', 'partial'
            'vision_unfreeze_layers': [9, 10, 11],  # Last 3 layers for fine-tuning when freeze_pattern='partial'
            'output_attentions': True,  # Output attention weights for interpretability
            'output_hidden_states': False,  # Whether to output hidden states from all layers
            
            # Language decoder settings (GPT-2 only)
            'gpt2_model_name': 'gpt2-medium',  # 'gpt2', 'gpt2-medium', 'gpt2-large', etc.
            'pretrained_decoder': True,  # Always use pretrained for better results
            'freeze_decoder': False,  # Allow training the decoder
            'decoder_hidden_dim': 1024,  # 1024 for gpt2-medium (adjust based on model)
            
            # Tokenizer settings
            'tokenizer_name': 'gpt2',  # Keep this consistent with gpt2_model_name
            'max_length': 256  # Maximum sequence length for training (includes input prompt)
        }
        
        # Training configuration
        self.train_config = {
            # Data settings
            'train_batch_size': 16,  # Reduced from 32 to account for larger models
            'val_batch_size': 16,
            'test_batch_size': 16,
            'num_workers': 5,  # Adjusted for better CPU utilization
            'seed': 42,
            'max_samples': 10000,  # Can increase if more data available
            'test_size': 0.1,
            'val_size': 0.1,

            # Optimizer settings
            'optimizer': 'adamw',
            'learning_rate': 5e-5,  # Slightly lower base LR for better stability
            'vision_learning_rate': 1e-5,  # Lower LR for vision encoder when unfrozen
            'weight_decay': 1e-4,  # Increased for better regularization
            'adam_epsilon': 1e-8,

            # Training settings
            'num_epochs': 50,  # Allow more training epochs
            'warmup_steps': 500,  # Adjusted for stability
            'gradient_accumulation_steps': 2,  # Help with larger batch sizes
            'max_grad_norm': 1.0,
            'early_stopping_patience': 3,  # Give more chances for improvement

            # Scheduler settings
            'scheduler': 'linear_warmup',
            'scheduler_factor': 0.5,
            'scheduler_patience': 2,

            # Logging & checkpointing
            'log_interval': 50,  # More frequent logging
            'eval_interval': 5,  # Evaluate each epoch
            'save_interval': 1  # Save each epoch
        }
        
        # Evaluation configuration
        self.eval_config = {
            'metrics': ['bleu', 'rouge', 'meteor', 'bertscore', 'chexbert'],  # Added CheXbert for medical accuracy
            'beam_size': 4,  # Standard beam size for medical text generation
            'max_length': 150,  # Typical chest X-ray report length (in tokens)
            'min_length': 15,  # Ensure reports have at least basic findings
            'no_repeat_ngram_size': 2,  # Smaller value for more natural repetition in medical context
            'early_stopping': True,
            'length_penalty': 0.8,  # Slight preference for conciseness in medical reports
            'num_return_sequences': 1,  # Return top sequence
            'top_p': 0.9,  # Use nucleus sampling for more focused medical text generation
            'do_sample': False,  # Use beam search for deterministic outputs
            'repetition_penalty': 1.2  # Reduce redundant language in reports
        }
        
        # Update configuration with command-line arguments
        if args is not None:
            self.update_from_args(args)
    
    def save(self, filepath):
        """
        Save configuration to a JSON file.
        
        Args:
            filepath (str): Path to save the configuration file.
        """
        config_dict = {
            'model_config': self.model_config,
            'train_config': self.train_config,
            'eval_config': self.eval_config
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    def load(self, filepath):
        """
        Load configuration from a JSON file.
        
        Args:
            filepath (str): Path to the configuration file.
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        self.model_config.update(config_dict.get('model_config', {}))
        self.train_config.update(config_dict.get('train_config', {}))
        self.eval_config.update(config_dict.get('eval_config', {}))
    
    def update_from_args(self, args):
        """
        Update configuration from command-line arguments.
        
        Args:
            args (argparse.Namespace): Command-line arguments.
        """
        # Convert args to dictionary
        args_dict = vars(args)
        
        # Update model config
        model_keys = set(self.model_config.keys())
        for key, value in args_dict.items():
            if key in model_keys and value is not None:
                self.model_config[key] = value
        
        # Update training config
        train_keys = set(self.train_config.keys())
        for key, value in args_dict.items():
            if key in train_keys and value is not None:
                self.train_config[key] = value
        
        # Update evaluation config
        eval_keys = set(self.eval_config.keys())
        for key, value in args_dict.items():
            if key in eval_keys and value is not None:
                self.eval_config[key] = value
        
        # Update paths if provided
        if args.checkpoint is not None:
            self.checkpoint_path = args.checkpoint
        
        if args.config_file is not None:
            self.load(args.config_file)
        
        # Update CSV file if provided
        if hasattr(args, 'csv_file') and args.csv_file is not None:
            self.csv_file = args.csv_file


def get_parser():
    """
    Get command-line argument parser.
    
    Returns:
        argparse.ArgumentParser: Argument parser.
    """
    parser = argparse.ArgumentParser(description='Radiology Report Generation')
    
    # General arguments
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'inference'],
                       default='train', help='Running mode')
    parser.add_argument('--stage', type=int, choices=[1, 2], default=1,
                       help='Training stage (1: frozen vision encoder, 2: fine-tuning vision encoder)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for resuming training or evaluation')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to configuration file')
    
    # Vision encoder arguments
    parser.add_argument('--vision_model_name', type=str,
                       help='Name of Vision Transformer model from Hugging Face')
    parser.add_argument('--freeze_vision', type=bool,
                       help='Whether to freeze the vision encoder')
    parser.add_argument('--vision_feature_selection', type=str, 
                       choices=['cls', 'all', 'mean', 'attention_weighted'],
                       help='Method to select vision features')
    parser.add_argument('--vision_freeze_pattern', type=str,
                       choices=['none', 'embeddings_only', 'partial'],
                       help='Pattern for selectively freezing vision encoder layers')
    parser.add_argument('--output_attentions', type=bool,
                       help='Whether to output attention weights for visualization')
    
    # Language decoder arguments
    parser.add_argument('--gpt2_model_name', type=str,
                       help='Name of GPT-2 model variant')
    parser.add_argument('--freeze_decoder', type=bool,
                       help='Whether to freeze the language decoder')
    parser.add_argument('--decoder_hidden_dim', type=int,
                       help='Hidden dimension for decoder/projection')
    
    # Training arguments
    parser.add_argument('--train_batch_size', type=int,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float,
                       help='Base learning rate')
    parser.add_argument('--vision_learning_rate', type=float,
                       help='Specific learning rate for the vision encoder during fine-tuning')
    parser.add_argument('--num_epochs', type=int,
                       help='Number of training epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int,
                       help='Number of steps to accumulate gradients')
    parser.add_argument('--seed', type=int,
                       help='Random seed')
    
    # Evaluation arguments
    parser.add_argument('--beam_size', type=int,
                       help='Beam size for generation')
    parser.add_argument('--max_length', type=int,
                       help='Maximum length of generated text')
    parser.add_argument('--min_length', type=int,
                       help='Minimum length of generated text')
    parser.add_argument('--length_penalty', type=float,
                       help='Length penalty for generation (>1 favors longer, <1 favors shorter)')
    parser.add_argument('--repetition_penalty', type=float,
                       help='Penalty for repeating tokens in generated text')
    
    return parser


def get_config():
    """
    Get configuration from command-line arguments.
    
    Returns:
        Config: Configuration object.
    """
    parser = get_parser()
    args = parser.parse_args()
    
    return Config(args) 
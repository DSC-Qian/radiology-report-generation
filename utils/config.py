import os
import json
import argparse


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
        self.data_path = os.path.join(self.project_root, '.')
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
            'encoder_type': 'resnet',  # 'resnet' or 'vit'
            'encoder_model': 'resnet50',  # For ResNet: 'resnet18', 'resnet50', etc.; For ViT: 'google/vit-base-patch16-224'
            'pretrained_encoder': True,
            'freeze_encoder': True,  # Stage 1: True, Stage 2: False
            
            # Mapping network settings
            'mapper_type': 'transformer',  # 'mlp' or 'transformer'
            'mapper_hidden_dim': 768,
            'mapper_num_layers': 2,
            'mapper_num_heads': 8,
            'mapper_seq_len': 16,
            'mapper_dropout': 0.1,
            
            # Language decoder settings
            'decoder_type': 'gpt2',  # 'gpt2' or 'biomedical'
            'decoder_model': 'gpt2',  # For GPT-2: 'gpt2', 'gpt2-medium', etc.; For Biomedical: 'microsoft/biogpt'
            'pretrained_decoder': True,
            'freeze_decoder': True,
            
            # Tokenizer settings
            'tokenizer_name': 'gpt2',
            'max_length': 512
        }
        
        # Training configuration
        self.train_config = {
            # Data settings
            'train_batch_size': 16,
            'val_batch_size': 16,
            'test_batch_size': 16,
            'num_workers': 4,
            'test_size': 0.1,
            'val_size': 0.1,
            'seed': 42,
            
            # Optimizer settings
            'optimizer': 'adam',  # 'adam' or 'adamw'
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'adam_epsilon': 1e-8,
            
            # Training settings
            'num_epochs': 50,
            'warmup_steps': 500,
            'gradient_accumulation_steps': 1,
            'max_grad_norm': 1.0,
            'early_stopping_patience': 5,
            
            # Scheduler settings
            'scheduler': 'linear_warmup',  # 'linear_warmup' or 'reduce_on_plateau'
            'scheduler_factor': 0.5,
            'scheduler_patience': 2,
            
            # Logging settings
            'log_interval': 100,
            'eval_interval': 1,  # Evaluate every n epochs
            'save_interval': 1,  # Save checkpoint every n epochs
        }
        
        # Evaluation configuration
        self.eval_config = {
            'metrics': ['bleu', 'rouge', 'meteor', 'bertscore'],
            'beam_size': 4,
            'max_length': 512,
            'min_length': 10,
            'no_repeat_ngram_size': 3,
            'early_stopping': True
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
                       help='Training stage (1: frozen encoder, 2: fine-tuning)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for resuming training or evaluation')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to configuration file')
    
    # Model arguments
    parser.add_argument('--encoder_type', type=str, choices=['resnet', 'vit'],
                       help='Type of vision encoder')
    parser.add_argument('--encoder_model', type=str,
                       help='Name of vision encoder model')
    parser.add_argument('--mapper_type', type=str, choices=['mlp', 'transformer'],
                       help='Type of mapping network')
    parser.add_argument('--decoder_type', type=str, choices=['gpt2', 'biomedical'],
                       help='Type of language decoder')
    parser.add_argument('--decoder_model', type=str,
                       help='Name of language decoder model')
    
    # Training arguments
    parser.add_argument('--train_batch_size', type=int,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int,
                       help='Number of training epochs')
    parser.add_argument('--seed', type=int,
                       help='Random seed')
    
    # Evaluation arguments
    parser.add_argument('--beam_size', type=int,
                       help='Beam size for generation')
    
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
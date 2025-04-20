import os
import argparse
import torch
import random
import numpy as np
from datetime import datetime

from utils.config import Config, get_parser
from training.trainer import train_model
from evaluation.evaluate import evaluate_model


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_output_dirs(config):
    """
    Create output directories if they don't exist.
    
    Args:
        config (Config): Configuration object.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)


def train(config, stage, resume_from=None):
    """
    Train the model.
    
    Args:
        config (Config): Configuration object.
        stage (int): Training stage (1 or 2).
        resume_from (str, optional): Path to checkpoint for resuming training.
        
    Returns:
        tuple: Best validation loss and metrics.
    """
    print(f"=== Starting Training (Stage {stage}) ===")
    create_output_dirs(config)
    
    # Set random seed
    set_seed(config.train_config['seed'])
    
    # Train model
    best_val_loss, best_val_metrics = train_model(config, stage, resume_from)
    
    print(f"=== Training Completed (Stage {stage}) ===")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return best_val_loss, best_val_metrics


def evaluate(config, checkpoint_path):
    """
    Evaluate the model.
    
    Args:
        config (Config): Configuration object.
        checkpoint_path (str): Path to the model checkpoint.
        
    Returns:
        dict: Evaluation metrics.
    """
    print(f"=== Starting Evaluation ===")
    create_output_dirs(config)
    
    # Set random seed
    set_seed(config.train_config['seed'])
    
    # Evaluate model
    metrics = evaluate_model(config, checkpoint_path)
    
    print(f"=== Evaluation Completed ===")
    
    return metrics


def main():
    """
    Main function to run training or evaluation.
    """
    # Parse command-line arguments
    parser = get_parser()
    parser.add_argument('--csv_file', type=str, default=None,
                       help='Path to the CSV file with image-report pairs. Overrides the default path.')
    args = parser.parse_args()
    
    # Create configuration
    config = Config(args)
    
    # Override CSV file path if provided
    if args.csv_file is not None:
        config.csv_file = args.csv_file
        print(f"Using custom CSV file: {config.csv_file}")
    
    # Create output directories
    create_output_dirs(config)
    
    # Run according to mode
    if args.mode == 'train':
        if args.stage == 1:
            # Stage 1: Train with frozen encoder
            train(config, stage=1, resume_from=args.checkpoint)
        elif args.stage == 2:
            # Stage 2: Fine-tune with unfrozen encoder
            if args.checkpoint is None:
                # If no checkpoint is provided, use the best model from stage 1
                stage1_checkpoint = os.path.join(config.checkpoint_dir, 'best_model_stage1.pt')
                if os.path.exists(stage1_checkpoint):
                    print(f"Using best model from stage 1: {stage1_checkpoint}")
                    train(config, stage=2, resume_from=stage1_checkpoint)
                else:
                    print(f"No checkpoint found for stage 1. Please provide a checkpoint.")
            else:
                train(config, stage=2, resume_from=args.checkpoint)
        else:
            raise ValueError(f"Invalid training stage: {args.stage}")
    
    elif args.mode == 'eval':
        if args.checkpoint is None:
            # If no checkpoint is provided, use the best model
            stage = args.stage or 1  # Default to stage 1 if not specified
            checkpoint_path = os.path.join(config.checkpoint_dir, f'best_model_stage{stage}.pt')
            if os.path.exists(checkpoint_path):
                print(f"Using best model from stage {stage}: {checkpoint_path}")
                evaluate(config, checkpoint_path)
            else:
                print(f"No checkpoint found for stage {stage}. Please provide a checkpoint.")
        else:
            evaluate(config, args.checkpoint)
    
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == '__main__':
    start_time = datetime.now()
    print(f"Starting at: {start_time}")
    
    main()
    
    end_time = datetime.now()
    print(f"Finished at: {end_time}")
    print(f"Elapsed time: {end_time - start_time}") 
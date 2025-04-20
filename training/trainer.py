import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.cuda.amp import autocast, GradScaler

from data.dataloader import get_dataloader
from models.report_generator import get_report_generator
from evaluation.metrics import compute_metrics


class Trainer:
    """
    Trainer class for training and evaluating the radiology report generation model.
    
    Args:
        config (Config): Configuration object with model, training, and evaluation parameters.
        stage (int): Training stage (1 or 2).
        resume_from (str, optional): Path to checkpoint for resuming training.
    """
    def __init__(self, config, stage=1, resume_from=None):
        self.config = config
        self.stage = stage
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default GPU memory optimization settings if not in config
        self.config.train_config.setdefault('use_mixed_precision', True)  # Enable mixed precision by default
        self.config.train_config.setdefault('train_batch_size', 4)  # Reduce default batch size
        self.config.train_config.setdefault('val_batch_size', 4)  # Reduce default batch size
        self.config.train_config.setdefault('gradient_accumulation_steps', 4)  # Increase gradient accumulation
        self.config.train_config.setdefault('max_length', 128)  # Ensure sequence length is limited
        
        # Set up mixed precision training
        self.scaler = GradScaler(enabled=self.config.train_config['use_mixed_precision'])
        
        # Update model config based on training stage
        if stage == 1:
            # Stage 1: Freeze encoder, train mapper and decoder
            self.config.model_config['freeze_encoder'] = True
        elif stage == 2:
            # Stage 2: Unfreeze encoder, fine-tune end-to-end
            self.config.model_config['freeze_encoder'] = False
        else:
            raise ValueError(f"Invalid training stage: {stage}")
        
        # Create model
        self.model = get_report_generator(self.config.model_config)
        self.model.to(self.device)
        
        # Print GPU memory information
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"Mixed precision training: {'Enabled' if self.config.train_config['use_mixed_precision'] else 'Disabled'}")
        
        # Print model's trainable parameters
        print("Trainable parameters:")
        total_params = 0
        trainable_params = 0
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
                print(f"  {name}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Create data loaders
        self.train_loader = get_dataloader(
            csv_file=self.config.csv_file,
            root_dir=self.config.data_path,
            batch_size=self.config.train_config['train_batch_size'],
            num_workers=self.config.train_config['num_workers'],
            max_length=self.config.model_config['max_length'],
            split='train',
            tokenizer_name=self.config.model_config['tokenizer_name'],
            test_size=self.config.train_config['test_size'],
            val_size=self.config.train_config['val_size'],
            seed=self.config.train_config['seed']
        )
        
        self.val_loader = get_dataloader(
            csv_file=self.config.csv_file,
            root_dir=self.config.data_path,
            batch_size=self.config.train_config['val_batch_size'],
            num_workers=self.config.train_config['num_workers'],
            max_length=self.config.model_config['max_length'],
            split='val',
            tokenizer_name=self.config.model_config['tokenizer_name'],
            test_size=self.config.train_config['test_size'],
            val_size=self.config.train_config['val_size'],
            seed=self.config.train_config['seed']
        )
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=os.path.join(self.config.log_dir, f"stage{stage}"))
        
        # Create optimizer
        # Get parameters that require gradients
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if not trainable_params:
            raise ValueError("No trainable parameters found in the model!")
        
        if self.config.train_config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                trainable_params,
                lr=self.config.train_config['learning_rate'],
                weight_decay=self.config.train_config['weight_decay']
            )
        elif self.config.train_config['optimizer'] == 'adamw':
            self.optimizer = AdamW(
                trainable_params,
                lr=self.config.train_config['learning_rate'],
                weight_decay=self.config.train_config['weight_decay'],
                eps=self.config.train_config['adam_epsilon']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.train_config['optimizer']}")
        
        # Create scheduler
        if self.config.train_config['scheduler'] == 'linear_warmup':
            # Calculate total training steps
            total_steps = len(self.train_loader) * self.config.train_config['num_epochs']
            
            # Create linear scheduler with warmup
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.train_config['warmup_steps'],
                num_training_steps=total_steps
            )
        elif self.config.train_config['scheduler'] == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.train_config['scheduler_factor'],
                patience=self.config.train_config['scheduler_patience'],
                verbose=True
            )
        else:
            self.scheduler = None
        
        # Initialize training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_metrics = None
        self.patience_counter = 0
        
        # Load checkpoint if provided
        if resume_from is not None:
            self.load_checkpoint(resume_from)
    
    def train(self):
        """
        Train the model for the specified number of epochs.
        
        Returns:
            tuple: Best validation loss and metrics.
        """
        print(f"Starting training (Stage {self.stage})...")
        print(f"Using device: {self.device}")
        print(f"Batch size: {self.config.train_config['train_batch_size']} (effective: {self.config.train_config['train_batch_size'] * self.config.train_config['gradient_accumulation_steps']})")
        print(f"Number of training samples: {len(self.train_loader.dataset)}")
        print(f"Number of validation samples: {len(self.val_loader.dataset)}")
        
        # Training loop
        for epoch in range(self.start_epoch, self.config.train_config['num_epochs']):
            print(f"Epoch {epoch+1}/{self.config.train_config['num_epochs']}")
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Evaluate on validation set
            if (epoch + 1) % self.config.train_config['eval_interval'] == 0:
                val_loss, val_metrics = self.evaluate()
                
                # Update scheduler if using reduce_on_plateau
                if self.config.train_config['scheduler'] == 'reduce_on_plateau':
                    self.scheduler.step(val_loss)
                
                # Save checkpoint if improved
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_val_metrics = val_metrics
                    self.patience_counter = 0
                    
                    # Save best model
                    self.save_checkpoint(
                        os.path.join(self.config.checkpoint_dir, f"best_model_stage{self.stage}.pt"),
                        is_best=True
                    )
                else:
                    self.patience_counter += 1
                
                # Log metrics
                self.writer.add_scalar('Loss/validation', val_loss, epoch)
                for metric_name, metric_value in val_metrics.items():
                    self.writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
                
                # Early stopping
                if self.patience_counter >= self.config.train_config['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config.train_config['save_interval'] == 0:
                self.save_checkpoint(
                    os.path.join(self.config.checkpoint_dir, f"checkpoint_stage{self.stage}_epoch{epoch+1}.pt")
                )
            
            # Log GPU memory usage after each epoch if available
            if torch.cuda.is_available():
                max_memory = torch.cuda.max_memory_allocated() / 1e9
                print(f"Max GPU memory used: {max_memory:.2f} GB")
                torch.cuda.reset_peak_memory_stats()
        
        # Close tensorboard writer
        self.writer.close()
        
        return self.best_val_loss, self.best_val_metrics
    
    def train_epoch(self):
        """
        Train the model for one epoch.
        
        Returns:
            float: Average training loss.
        """
        self.model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        # Zero gradients at the start of the epoch
        self.optimizer.zero_grad()
        
        # Progress bar
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Set labels for autoregressive training (shift input_ids right)
            # For GPT models, typically labels are the same as input_ids
            # The model handles masking internally
            labels = input_ids.clone()
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.train_config['use_mixed_precision']):
                outputs = self.model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Check if loss is a tensor and requires grad
                if not isinstance(outputs['loss'], torch.Tensor):
                    raise TypeError(f"Expected loss to be a tensor, got {type(outputs['loss'])}")
                
                loss = outputs['loss']
                
                # Scale loss for gradient accumulation
                if self.config.train_config['gradient_accumulation_steps'] > 1:
                    loss = loss / self.config.train_config['gradient_accumulation_steps']
            
            # Backward pass with mixed precision
            self.scaler.scale(loss).backward()
            
            # Update weights if gradient accumulation steps are reached
            if (batch_idx + 1) % self.config.train_config['gradient_accumulation_steps'] == 0:
                # Clip gradients with mixed precision
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.train_config['max_grad_norm']
                )
                
                # Update weights with mixed precision
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Update scheduler if using linear warmup
                if self.config.train_config['scheduler'] == 'linear_warmup':
                    self.scheduler.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Update global step
                self.global_step += 1
            
            # Update progress bar
            epoch_loss += loss.item() * self.config.train_config['gradient_accumulation_steps']
            progress_bar.set_postfix({
                'loss': epoch_loss / (batch_idx + 1)
            })
            
            # Log to tensorboard
            if (self.global_step + 1) % self.config.train_config['log_interval'] == 0:
                self.writer.add_scalar(
                    'Loss/train',
                    loss.item() * self.config.train_config['gradient_accumulation_steps'],
                    self.global_step
                )
                
            # Clear GPU cache periodically if memory is tight
            if (batch_idx + 1) % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate average loss
        epoch_loss /= len(self.train_loader)
        
        # Log epoch metrics
        elapsed_time = time.time() - start_time
        print(f"Epoch finished in {elapsed_time:.2f}s. Average loss: {epoch_loss:.4f}")
        
        return epoch_loss
    
    def evaluate(self):
        """
        Evaluate the model on the validation set.
        
        Returns:
            tuple: Validation loss and metrics.
        """
        self.model.eval()
        val_loss = 0.0
        
        # Lists to store reference and hypothesis texts
        references = []
        hypotheses = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Set labels for autoregressive training (shift input_ids right)
                labels = input_ids.clone()
                
                # Forward pass with mixed precision
                with autocast(enabled=self.config.train_config['use_mixed_precision']):
                    outputs = self.model(
                        images=images,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs['loss']
                
                val_loss += loss.item()
                
                # Generate text for metrics - use smaller batch size if needed
                max_gen_batch_size = max(1, self.config.train_config['val_batch_size'] // 2)
                for i in range(0, len(images), max_gen_batch_size):
                    batch_images = images[i:i+max_gen_batch_size]
                    with autocast(enabled=self.config.train_config['use_mixed_precision']):
                        batch_generated_ids = self.model.generate(batch_images)
                    
                    # Add generated texts to the list
                    for j in range(len(batch_images)):
                        idx = i + j
                        if idx < len(images):  # Safety check
                            # Reference text
                            reference = batch['report_text'][idx]
                            references.append(reference)
                            
                            # Generated text
                            hypothesis = self.model.tokenizer.decode(
                                batch_generated_ids[j],
                                skip_special_tokens=True
                            )
                            hypotheses.append(hypothesis)
                
                # Clear GPU cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Calculate average loss
        val_loss /= len(self.val_loader)
        
        # Compute evaluation metrics
        metrics = compute_metrics(references, hypotheses, self.config.eval_config['metrics'])
        
        # Log metrics
        print(f"Validation loss: {val_loss:.4f}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        
        return val_loss, metrics
    
    def save_checkpoint(self, filepath, is_best=False):
        """
        Save a checkpoint of the model.
        
        Args:
            filepath (str): Path to save the checkpoint.
            is_best (bool): Whether this is the best model so far.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),  # Save mixed precision scaler state
            'best_val_loss': self.best_val_loss,
            'best_val_metrics': self.best_val_metrics,
            'epoch': self.start_epoch,
            'global_step': self.global_step,
            'patience_counter': self.patience_counter,
            'config': {
                'model_config': self.config.model_config,
                'train_config': self.config.train_config,
                'eval_config': self.config.eval_config
            }
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            print(f"Saving best model with validation loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, filepath):
        """
        Load a checkpoint.
        
        Args:
            filepath (str): Path to the checkpoint.
        """
        if not os.path.exists(filepath):
            print(f"Checkpoint not found: {filepath}")
            return
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if exists
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load mixed precision scaler state if exists
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_metrics = checkpoint['best_val_metrics']
        self.patience_counter = checkpoint['patience_counter']
        
        print(f"Resumed from checkpoint: {filepath}")
        print(f"Starting from epoch: {self.start_epoch}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def train_model(config, stage=1, resume_from=None):
    """
    Train the model.
    
    Args:
        config (Config): Configuration object.
        stage (int): Training stage (1 or 2).
        resume_from (str, optional): Path to checkpoint for resuming training.
        
    Returns:
        tuple: Best validation loss and metrics.
    """
    # Set environment variables for better GPU memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    trainer = Trainer(config, stage, resume_from)
    return trainer.train() 
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from transformers import get_linear_schedule_with_warmup
from torch.optim   import AdamW 

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
        
        # Print model's trainable parameters
        print("Trainable parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"  {name}")
        
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
            seed=self.config.train_config['seed'],
            max_samples=self.config.train_config['max_samples']
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
            seed=self.config.train_config['seed'],
            max_samples=1000
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
        self.best_combined_score = -float('inf')  # For combined metrics (higher is better)
        self.best_val_metrics = None
        self.patience_counter = 0
        
        # Load checkpoint if provided
        if resume_from is not None:
            self.load_checkpoint(resume_from)
    
    def calculate_combined_score(self, metrics, val_loss):
        """
        Calculate a combined score based on multiple metrics for early stopping.
        
        Args:
            metrics (dict): Dictionary of evaluation metrics.
            val_loss (float): Validation loss.
            
        Returns:
            float: Combined score (higher is better).
        """
        # Define weights for different metric categories
        clinical_weight = 0.6  # Higher weight for clinical accuracy
        nlp_weight = 0.3       # Weight for NLP metrics
        loss_weight = 0.1      # Small weight for validation loss
        
        # Initialize scores
        clinical_score = 0.0
        nlp_score = 0.0
        
        # Calculate clinical metrics score (if available)
        clinical_metrics = []
        if 'chexbert' in metrics:
            # If CheXbert results are available
            if isinstance(metrics['chexbert'], dict):
                # Average F1 scores across all conditions
                f1_scores = [v.get('f1', 0.0) for k, v in metrics['chexbert'].items() 
                            if isinstance(v, dict) and 'f1' in v]
                if f1_scores:
                    clinical_metrics.append(np.mean(f1_scores))
        
        # Add RadGraph scores if available
        if 'radgraph' in metrics and isinstance(metrics['radgraph'], dict):
            if 'f1' in metrics['radgraph']:
                clinical_metrics.append(metrics['radgraph']['f1'])
        
        # Add entity recognition scores if available
        if 'entity_f1' in metrics:
            clinical_metrics.append(metrics['entity_f1'])
        
        # Calculate clinical score
        if clinical_metrics:
            clinical_score = np.mean(clinical_metrics)
        
        # Calculate NLP metrics score
        nlp_metrics = []
        
        # ROUGE-L (typically under 'rouge' key with 'l' subkey)
        if 'rouge' in metrics and isinstance(metrics['rouge'], dict) and 'l' in metrics['rouge']:
            if isinstance(metrics['rouge']['l'], dict) and 'f' in metrics['rouge']['l']:
                nlp_metrics.append(metrics['rouge']['l']['f'])
            elif isinstance(metrics['rouge']['l'], (int, float)):
                nlp_metrics.append(metrics['rouge']['l'])
        
        # BLEU score
        if 'bleu' in metrics:
            nlp_metrics.append(metrics['bleu'])
        
        # BERTScore
        if 'bertscore' in metrics and isinstance(metrics['bertscore'], dict) and 'f1' in metrics['bertscore']:
            nlp_metrics.append(metrics['bertscore']['f1'])
        
        # Calculate NLP score
        if nlp_metrics:
            nlp_score = np.mean(nlp_metrics)
        
        # Normalize validation loss to a 0-1 scale where higher is better
        # Assuming validation loss is typically in range 0-10
        loss_score = max(0, 1 - (val_loss / 10.0))
        
        # Calculate combined score
        combined_score = (
            clinical_weight * clinical_score + 
            nlp_weight * nlp_score + 
            loss_weight * loss_score
        )
        
        # Log the component scores
        print(f"Combined Score Components - Clinical: {clinical_score:.4f}, NLP: {nlp_score:.4f}, Loss: {loss_score:.4f}")
        print(f"Combined Score: {combined_score:.4f}")
        
        return combined_score
    
    def train(self):
        """
        Train the model for the specified number of epochs.
        
        Returns:
            tuple: Best validation loss and metrics.
        """
        print(f"Starting training (Stage {self.stage})...")
        print(f"Using device: {self.device}")
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
                
                # Calculate combined score for early stopping
                combined_score = self.calculate_combined_score(val_metrics, val_loss)
                
                # Update scheduler if using reduce_on_plateau
                if self.config.train_config['scheduler'] == 'reduce_on_plateau':
                    self.scheduler.step(val_loss)
                
                # Check if model improved based on combined score
                improved = False
                if combined_score > self.best_combined_score:
                    print(f"Combined score improved from {self.best_combined_score:.4f} to {combined_score:.4f}")
                    self.best_combined_score = combined_score
                    improved = True
                    
                # Also track validation loss improvement for compatibility
                if val_loss < self.best_val_loss:
                    print(f"Validation loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}")
                    self.best_val_loss = val_loss
                    
                # Save checkpoint if improved on combined score
                if improved:
                    self.best_val_metrics = val_metrics
                    self.patience_counter = 0
                    
                    # Save best model
                    self.save_checkpoint(
                        os.path.join(self.config.checkpoint_dir, f"best_model_stage{self.stage}.pt"),
                        is_best=True
                    )
                else:
                    self.patience_counter += 1
                    print(f"No improvement in combined score. Patience counter: {self.patience_counter}/{self.config.train_config['early_stopping_patience']}")
                
                # Log metrics
                self.writer.add_scalar('Loss/validation', val_loss, epoch)
                self.writer.add_scalar('Metrics/combined_score', combined_score, epoch)
                
                for metric_name, metric_value in val_metrics.items():
                    # Handle different metric types for TensorBoard logging
                    if isinstance(metric_value, dict):
                        # For dictionary metrics, log each key-value pair separately
                        for k, v in metric_value.items():
                            if isinstance(v, (int, float)):
                                self.writer.add_scalar(f'Metrics/{metric_name}/{k}', v, epoch)
                            # Skip string values for TensorBoard (it only accepts numeric values)
                    elif isinstance(metric_value, (int, float)):
                        # For scalar metrics, log directly
                        self.writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
                    # Skip string values for TensorBoard (it only accepts numeric values)
                
                # Early stopping based on patience
                if self.patience_counter >= self.config.train_config['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch+1}. No improvement in combined score for {self.patience_counter} evaluations.")
                    break
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config.train_config['save_interval'] == 0:
                self.save_checkpoint(
                    os.path.join(self.config.checkpoint_dir, f"checkpoint_stage{self.stage}_epoch{epoch+1}.pt")
                )
        
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
            
            # Forward pass
            outputs = self.model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Check if loss is a tensor and requires grad
            if not isinstance(outputs['loss'], torch.Tensor):
                raise TypeError(f"Expected loss to be a tensor, got {type(outputs['loss'])}")
            
            if not outputs['loss'].requires_grad:
                # Create a gradient-enabled wrapper for the loss
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                if trainable_params:
                    # Connect the loss to trainable parameters to ensure gradient flow
                    loss = outputs['loss'] + 0 * sum(p.sum() for p in trainable_params)
                else:
                    print("Warning: No trainable parameters found. Check model configuration.")
                    loss = outputs['loss']
            else:
                loss = outputs['loss']
            
            # Backward pass
            if self.config.train_config['gradient_accumulation_steps'] > 1:
                loss = loss / self.config.train_config['gradient_accumulation_steps']
            
            loss.backward()
            
            # Update weights if gradient accumulation steps are reached
            if (batch_idx + 1) % self.config.train_config['gradient_accumulation_steps'] == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.train_config['max_grad_norm']
                )
                
                # Update weights
                self.optimizer.step()
                
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
            tuple: Validation loss and metrics dictionary.
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
                
                # Forward pass
                outputs = self.model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                val_loss += loss.item()
                del outputs, loss
                
                # Generate text for metrics
                generated_texts = self.model.generate(images)
                
                # Store reference and generated texts
                for i in range(len(images)):
                    # Reference text
                    reference = batch['report_text'][i]
                    references.append(reference)
                    
                    # Generated text (already decoded in model.generate)
                    hypothesis = generated_texts[i]
                    hypotheses.append(hypothesis)

        torch.cuda.empty_cache()
        
        # Calculate average loss
        val_loss /= len(self.val_loader)
        
        # Compute evaluation metrics
        metrics = compute_metrics(references, hypotheses, self.config.eval_config['metrics'])
        
        # Log metrics
        print(f"Validation loss: {val_loss:.4f}")
        for metric_name, metric_value in metrics.items():
            # Print out metrics, handling different value types
            if isinstance(metric_value, dict):
                print(f"{metric_name}:")
                for k, v in metric_value.items():
                    if isinstance(v, (int, float)):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
            elif isinstance(metric_value, (int, float)):
                print(f"{metric_name}: {metric_value:.4f}")
            else:
                print(f"{metric_name}: {metric_value}")
        
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
            'best_val_loss': self.best_val_loss,
            'best_combined_score': self.best_combined_score,
            'best_val_metrics': self.best_val_metrics,
            'epoch': self.start_epoch,
            'global_step': self.global_step,
            'patience_counter': self.patience_counter,
            'stage': self.stage,
            'config': {
                'model_config': self.config.model_config,
                'train_config': self.config.train_config,
                'eval_config': self.config.eval_config
            }
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            print(f"Saving best model with combined score: {self.best_combined_score:.4f}, validation loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, filepath):
        """
        Load a checkpoint.
        
        Args:
            filepath (str): Path to the checkpoint.
        """
        if not os.path.exists(filepath):
            print(f"Checkpoint not found: {filepath}")
            return
        
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Check if the stage in checkpoint matches current stage
        checkpoint_stage = checkpoint.get('config', {}).get('model_config', {}).get('freeze_encoder', None)
        current_stage = self.config.model_config.get('freeze_encoder', None)
        
        # If stages are different (e.g., moving from stage 1 to stage 2), don't load optimizer/scheduler
        is_stage_transition = checkpoint_stage is not None and current_stage is not None and checkpoint_stage != current_stage
        
        if not is_stage_transition:
            # Only load optimizer and scheduler if not transitioning between stages
            try:
                # Load optimizer state
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Load scheduler state if exists
                if checkpoint['scheduler_state_dict'] and self.scheduler:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    
                # Load training state
                self.start_epoch = checkpoint['epoch'] + 1
                self.global_step = checkpoint['global_step']
                self.best_val_loss = checkpoint['best_val_loss']
                self.best_combined_score = checkpoint.get('best_combined_score', -float('inf'))  # Default if not present
                self.best_val_metrics = checkpoint['best_val_metrics']
                self.patience_counter = checkpoint['patience_counter']
                
                print(f"Resumed from checkpoint: {filepath}")
                print(f"Starting from epoch: {self.start_epoch}")
                print(f"Best validation loss: {self.best_val_loss:.4f}")
                print(f"Best combined score: {self.best_combined_score:.4f}")
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not load optimizer/scheduler states: {e}")
                print("Initializing new optimizer and scheduler while keeping model weights.")
                self.start_epoch = 0
                self.global_step = 0
                self.best_val_loss = float('inf')
                self.best_combined_score = -float('inf')
                self.best_val_metrics = None
                self.patience_counter = 0
        else:
            # When transitioning between stages, initialize fresh training state
            print(f"Detected stage transition. Loading only model weights from checkpoint: {filepath}")
            print("Initializing new optimizer and scheduler for the new stage.")
            self.start_epoch = 0
            self.global_step = 0
            self.best_val_loss = float('inf')
            self.best_combined_score = -float('inf')
            self.best_val_metrics = None
            self.patience_counter = 0


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
    trainer = Trainer(config, stage, resume_from)
    return trainer.train() 
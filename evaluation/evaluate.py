import os
import torch
import json
import pandas as pd
from tqdm import tqdm
import logging
import traceback
import numpy as np

from data.dataloader import get_dataloader
from models.report_generator import get_report_generator
from evaluation.metrics import compute_metrics
from utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(config, checkpoint_path):
    """
    Evaluate a trained model on the test set.
    
    Args:
        config (Config): Configuration object.
        checkpoint_path (str): Path to the model checkpoint.
        
    Returns:
        dict: Evaluation metrics.
    """
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Ensure checkpoint path exists
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found at {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        # Get evaluation config parameters
        eval_batch_size = config.eval_config.get('batch_size', config.train_config.get('test_batch_size', 8))
        num_workers = config.eval_config.get('num_workers', config.train_config.get('num_workers', 4))
        samples_to_evaluate = config.eval_config.get('samples_to_evaluate', 1000)
        use_amp = config.eval_config.get('use_amp', True) and torch.cuda.is_available()  # Use mixed precision if available
        
        # Create test dataloader
        logger.info("Creating test dataloader...")
        test_loader = get_dataloader(
            csv_file=config.csv_file,
            root_dir=config.data_path,
            batch_size=eval_batch_size,  # Use evaluation batch size
            num_workers=num_workers,     # Use evaluation num_workers
            max_length=config.model_config['max_length'],
            split='test',
            tokenizer_name=config.model_config['tokenizer_name'],
            test_size=config.train_config['test_size'],
            val_size=config.train_config['val_size'],
            seed=config.train_config['seed']
        )
        
        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        try:
            # Explicitly set weights_only=False if you trust the checkpoint source
            # This is often needed for checkpoints saved with older PyTorch versions or containing arbitrary objects
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            logger.info("Checkpoint loaded successfully with weights_only=False.")
        except Exception as e:
            logger.error(f"Failed to load checkpoint with weights_only=False: {e}")
            logger.error(traceback.format_exc())
            # If loading fails even with weights_only=False, re-raise the error
            # as we cannot proceed without a valid checkpoint.
            raise RuntimeError(f"Could not load checkpoint from {checkpoint_path}") from e
        
        # Get model configuration from checkpoint
        model_config = checkpoint.get('config', {}).get('model_config', config.model_config)
        
        # Create model
        logger.info("Creating model...")
        model = get_report_generator(model_config)
        model.to(device)
        
        # Load model weights
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            # Try with a more permissive loading strategy
            logger.warning(f"Standard state_dict loading failed: {e}. Trying with strict=False...")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Set model to evaluation mode
        model.eval()
        
        # Initialize automatic mixed precision scaler if using AMP
        if use_amp:
            logger.info("Using automatic mixed precision for inference")
            amp_scaler = torch.cuda.amp.GradScaler()
        else:
            amp_scaler = None
        
        # Lists to store reference and hypothesis texts, image paths, and report paths
        references = []
        hypotheses = []
        image_paths = []
        report_paths = []
        evaluated_count = 0  # Initialize counter
        
        logger.info(f"Starting evaluation on {samples_to_evaluate} samples with batch size {eval_batch_size}...")
        
        # Generate reports and collect metrics using inference_mode for speed
        with torch.inference_mode():
            for batch in tqdm(test_loader, desc=f"Evaluation (first {samples_to_evaluate} samples)"):
                if evaluated_count >= samples_to_evaluate:
                    break # Stop if we already have enough samples

                try:
                    # Move batch to device
                    images = batch['image'].to(device)
                    
                    # Generate reports with optional AMP
                    if use_amp and hasattr(torch.cuda.amp, 'autocast'):
                        with torch.cuda.amp.autocast():
                            generated_reports = model.generate(images)
                    else:
                        generated_reports = model.generate(images)
                    
                    # Process results
                    current_batch_size = len(images)
                    remaining_needed = samples_to_evaluate - evaluated_count
                    
                    # Determine how many samples to take from this batch
                    num_to_take = min(current_batch_size, remaining_needed)
                    
                    for i in range(num_to_take):
                        references.append(batch['report_text'][i])
                        hypotheses.append(generated_reports[i])
                        image_paths.append(batch['image_path'][i])
                        report_paths.append(batch['report_path'][i])
                    
                    evaluated_count += num_to_take # Update counter
                    
                    # Clear CUDA cache periodically to avoid memory issues
                    if torch.cuda.is_available() and evaluated_count % 50 == 0:
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    logger.error(traceback.format_exc())
                    continue  # Skip this batch and continue with the next one
        
        # --- Check if any samples were evaluated ---
        if evaluated_count == 0:
            logger.error("No samples were evaluated. Check dataloader or evaluation limit.")
            return {"error": "No samples were evaluated"} # Return error metrics
        
        logger.info(f"Successfully evaluated {evaluated_count} samples")

        # Compute evaluation metrics
        logger.info(f"Computing metrics on {evaluated_count} samples...")
        # This compute_metrics function now handles all the NLP metrics (ROUGE, BERTScore, etc.)
        metrics = compute_metrics(references, hypotheses)

        # --- SAVE GENERATED REPORTS TO DISK ---
        # Ensure the results directory exists (configured in utils.config)
        os.makedirs(config.result_dir, exist_ok=True)
        reports_path = os.path.join(config.result_dir, 'generated_reports.txt')
        
        logger.info(f"Saving generated reports to {reports_path}...")
        with open(reports_path, 'w', encoding='utf-8') as f:
            for idx, (ref, hyp) in enumerate(zip(references, hypotheses), start=1):
                f.write(f"### Example {idx}\n")
                f.write("Reference:\n")
                f.write(ref.replace('\n', ' ') + "\n\n")
                f.write("Generated:\n")
                f.write(hyp.replace('\n', ' ') + "\n\n---\n\n")
        logger.info(f"Saved {len(hypotheses)} generated reports to {reports_path}")
        
        # Save outputs
        results = {
            'metrics': metrics,
            'examples': []
        }
        
        # Save examples for qualitative analysis
        for i in range(min(10, len(references))):
            results['examples'].append({
                'image_path': image_paths[i],
                'report_path': report_paths[i],
                'reference': references[i],
                'hypothesis': hypotheses[i]
            })
        
        # Print metrics
        logger.info("Evaluation Metrics:")
        if metrics:
            for metric_name, metric_value in metrics.items():
                # Check if value is a float or can be formatted as such
                if isinstance(metric_value, (float, np.float32, np.float64)):
                    logger.info(f"{metric_name}: {metric_value:.4f}")
                else:
                     logger.info(f"{metric_name}: {metric_value}") # Print non-float values directly
        else:
            logger.warning("No metrics were computed or returned.")
        
        # Save results to file
        results_path = os.path.join(config.result_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save predictions to CSV
        predictions_df = pd.DataFrame({
            'image_path': image_paths,
            'report_path': report_paths,
            'reference': references,
            'hypothesis': hypotheses
        })
        predictions_path = os.path.join(config.result_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        
        logger.info(f"Evaluation completed successfully. Results saved to {config.result_dir}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def evaluate_batch(config, checkpoint_path, batch_size=4):
    """
    Evaluate model on a single batch of data.
    Used for quick evaluation during development.
    
    Args:
        config (Config): Configuration object.
        checkpoint_path (str): Path to the model checkpoint.
        batch_size (int): Batch size for evaluation.
        
    Returns:
        dict: Evaluation metrics.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Get evaluation config parameters
    num_workers = config.eval_config.get('num_workers', 1)
    use_amp = config.eval_config.get('use_amp', True) and torch.cuda.is_available()
    
    # Create test dataloader with a small batch size
    logger.info("Creating test dataloader...")
    test_loader = get_dataloader(
        csv_file=config.csv_file,
        root_dir=config.data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=config.model_config['max_length'],
        split='test',
        tokenizer_name=config.model_config['tokenizer_name'],
        test_size=config.train_config['test_size'],
        val_size=config.train_config['val_size'],
        seed=config.train_config['seed']
    )
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model configuration from checkpoint
    model_config = checkpoint.get('config', {}).get('model_config', config.model_config)
    
    # Create model
    logger.info("Creating model...")
    model = get_report_generator(model_config)
    model.to(device)
    
    # Load model weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        logger.warning(f"Standard state_dict loading failed: {e}. Trying with strict=False...")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Set model to evaluation mode
    model.eval()
    
    # Lists to store reference and hypothesis texts
    references = []
    hypotheses = []
    
    # Generate reports and collect metrics
    logger.info("Generating reports for one batch...")
    with torch.inference_mode():
        batch = next(iter(test_loader))
        
        # Move batch to device
        images = batch['image'].to(device)
        
        # Generate reports with optional AMP
        if use_amp and hasattr(torch.cuda.amp, 'autocast'):
            with torch.cuda.amp.autocast():
                generated_reports = model.generate(images)
        else:
            generated_reports = model.generate(images)
        
        # Collect reference and hypothesis texts
        for i in range(len(images)):
            references.append(batch['report_text'][i])
            hypotheses.append(generated_reports[i])
    
    # Compute evaluation metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(references, hypotheses)
    
    # Print metrics
    logger.info("Evaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (float, np.float32, np.float64)):
            logger.info(f"{metric_name}: {metric_value:.4f}")
        else:
            logger.info(f"{metric_name}: {metric_value}")
    
    # Print examples
    for i in range(len(references)):
        logger.info(f"\nExample {i+1}:")
        logger.info(f"Reference: {references[i]}")
        logger.info(f"Generated: {hypotheses[i]}")
    
    return metrics

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Radiology Report Generation Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--config_file', type=str, default=None,
                      help='Path to the configuration file')
    parser.add_argument('--batch', action='store_true',
                      help='Run evaluation on a small batch (for testing)')
    parser.add_argument('--samples', type=int, default=100,
                      help='Number of samples to evaluate (default: 100)')
    parser.add_argument('--batch_size', type=int, default=None,
                      help='Batch size for evaluation (default: use config)')
    parser.add_argument('--num_workers', type=int, default=None,
                      help='Number of workers for data loading (default: use config)')
    parser.add_argument('--no_amp', action='store_true',
                      help='Disable automatic mixed precision')
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    if args.config_file:
        config.load(args.config_file)
    
    # Ensure eval_config exists
    if not hasattr(config, 'eval_config'):
        config.eval_config = {}
        
    # Set evaluation parameters from command line if specified
    if args.samples:
        config.eval_config['samples_to_evaluate'] = args.samples
    
    if args.batch_size:
        config.eval_config['batch_size'] = args.batch_size
        
    if args.num_workers is not None:
        config.eval_config['num_workers'] = args.num_workers
        
    if args.no_amp:
        config.eval_config['use_amp'] = False
    
    # Run evaluation
    if args.batch:
        evaluate_batch(config, args.checkpoint)
    else:
        evaluate_model(config, args.checkpoint) 
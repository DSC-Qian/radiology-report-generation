import os
import torch
import json
import pandas as pd
from tqdm import tqdm

from data.dataloader import get_dataloader
from models.report_generator import get_report_generator
from evaluation.metrics import compute_metrics
from utils.config import Config


def evaluate_model(config, checkpoint_path):
    """
    Evaluate a trained model on the test set.
    
    Args:
        config (Config): Configuration object.
        checkpoint_path (str): Path to the model checkpoint.
        
    Returns:
        dict: Evaluation metrics.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test dataloader
    test_loader = get_dataloader(
        csv_file=config.csv_file,
        root_dir=config.data_path,
        batch_size=config.train_config['test_batch_size'],
        num_workers=config.train_config['num_workers'],
        max_length=config.model_config['max_length'],
        split='test',
        tokenizer_name=config.model_config['tokenizer_name'],
        test_size=config.train_config['test_size'],
        val_size=config.train_config['val_size'],
        seed=config.train_config['seed']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration from checkpoint
    model_config = checkpoint.get('config', {}).get('model_config', config.model_config)
    
    # Create model
    model = get_report_generator(model_config)
    model.to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
    # Lists to store reference and hypothesis texts, image paths, and report paths
    references = []
    hypotheses = []
    image_paths = []
    report_paths = []
    
    # Generate reports and collect metrics
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluation"):
            # Move batch to device
            images = batch['image'].to(device)
            
            # Generate reports
            generated_reports = model.generate(images)
            
            # Collect reference and hypothesis texts
            for i in range(len(images)):
                references.append(batch['report_text'][i])
                hypotheses.append(generated_reports[i])
                image_paths.append(batch['image_path'][i])
                report_paths.append(batch['report_path'][i])
    
    # Compute evaluation metrics
    metrics = compute_metrics(references, hypotheses, config.eval_config['metrics'])
    
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
    print("Evaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
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
    
    return metrics


def evaluate_batch(config, checkpoint_path, batch_size=4):
    """
    Evaluate a trained model on a small batch of examples.
    Useful for quick testing and debugging.
    
    Args:
        config (Config): Configuration object.
        checkpoint_path (str): Path to the model checkpoint.
        batch_size (int): Number of examples to evaluate.
        
    Returns:
        dict: Evaluation metrics.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test dataloader with small batch size
    test_loader = get_dataloader(
        csv_file=config.csv_file,
        root_dir=config.data_path,
        batch_size=batch_size,
        num_workers=1,
        max_length=config.model_config['max_length'],
        split='test',
        tokenizer_name=config.model_config['tokenizer_name'],
        test_size=config.train_config['test_size'],
        val_size=config.train_config['val_size'],
        seed=config.train_config['seed']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration from checkpoint
    model_config = checkpoint.get('config', {}).get('model_config', config.model_config)
    
    # Create model
    model = get_report_generator(model_config)
    model.to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
    # Get a single batch for evaluation
    batch = next(iter(test_loader))
    
    # Move batch to device
    images = batch['image'].to(device)
    
    # Generate reports
    with torch.no_grad():
        generated_reports = model.generate(images)
    
    # Collect reference and hypothesis texts
    references = batch['report_text']
    hypotheses = generated_reports
    
    # Compute evaluation metrics
    metrics = compute_metrics(references, hypotheses, config.eval_config['metrics'])
    
    # Print examples
    print("\nEvaluation Examples:")
    for i in range(len(references)):
        print(f"\nExample {i+1}:")
        print(f"Image path: {batch['image_path'][i]}")
        print(f"Reference: {references[i][:100]}...")
        print(f"Generated: {hypotheses[i][:100]}...")
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
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
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    if args.config_file:
        config.load(args.config_file)
    
    # Run evaluation
    if args.batch:
        evaluate_batch(config, args.checkpoint)
    else:
        evaluate_model(config, args.checkpoint) 
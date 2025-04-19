import os
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from models.report_generator import get_report_generator
from utils.config import Config


def load_model(checkpoint_path, device):
    """
    Load a trained model from a checkpoint.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint.
        device (torch.device): Device to load the model to.
        
    Returns:
        RadiologyReportGenerator: Loaded model.
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration from checkpoint
    model_config = checkpoint.get('config', {}).get('model_config', {})
    
    # Create model
    model = get_report_generator(model_config)
    model.to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
    return model


def preprocess_image(image_path, transform=None):
    """
    Preprocess an image for inference.
    
    Args:
        image_path (str): Path to the image.
        transform (callable, optional): Transform to apply to the image.
        
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # If no transform provided, use default
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    # Apply transform
    image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def generate_report(model, image_tensor, device):
    """
    Generate a radiology report for an image.
    
    Args:
        model (RadiologyReportGenerator): Trained model.
        image_tensor (torch.Tensor): Preprocessed image tensor.
        device (torch.device): Device to run inference on.
        
    Returns:
        str: Generated report.
    """
    # Move image to device
    image_tensor = image_tensor.to(device)
    
    # Generate report
    with torch.no_grad():
        generated_reports = model.generate(image_tensor)
    
    return generated_reports[0]


def visualize_results(image_path, report):
    """
    Visualize the image and generated report.
    
    Args:
        image_path (str): Path to the input image.
        report (str): Generated report text.
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Chest X-ray Image')
    plt.axis('off')
    
    # Display report
    plt.subplot(1, 2, 2)
    plt.text(0.1, 0.5, report, fontsize=12, wrap=True)
    plt.title('Generated Radiology Report')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def main(args):
    """
    Main function for inference.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create configuration
    config = Config()
    if args.config_file:
        config.load(args.config_file)
    
    # Load model
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    # Preprocess image
    print(f"Processing image: {args.image}")
    image_tensor = preprocess_image(args.image)
    
    # Generate report
    print("Generating report...")
    report = generate_report(model, image_tensor, device)
    
    # Print report
    print("\nGenerated Report:")
    print("-" * 80)
    print(report)
    print("-" * 80)
    
    # Save report to file if required
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to: {args.output}")
    
    # Visualize results if required
    if args.visualize:
        visualize_results(args.image, report)


def batch_inference(args):
    """
    Run inference on a batch of images.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create configuration
    config = Config()
    if args.config_file:
        config.load(args.config_file)
    
    # Load model
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    # Get all image files in the directory
    if os.path.isdir(args.image_dir):
        image_files = [
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))
        ]
    else:
        raise ValueError(f"Directory not found: {args.image_dir}")
    
    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each image
    print(f"Processing {len(image_files)} images...")
    for image_file in image_files:
        print(f"Processing: {image_file}")
        
        # Preprocess image
        image_tensor = preprocess_image(image_file)
        
        # Generate report
        report = generate_report(model, image_tensor, device)
        
        # Save report to file
        if args.output_dir:
            # Create output path
            base_name = os.path.splitext(os.path.basename(image_file))[0]
            output_path = os.path.join(args.output_dir, f"{base_name}_report.txt")
            
            # Save report
            with open(output_path, 'w') as f:
                f.write(report)
            
            print(f"Report saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for Radiology Report Generation')
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Inference mode')
    
    # Single image inference
    single_parser = subparsers.add_parser('single', help='Single image inference')
    single_parser.add_argument('--image', type=str, required=True,
                              help='Path to the input image')
    single_parser.add_argument('--checkpoint', type=str, required=True,
                              help='Path to the model checkpoint')
    single_parser.add_argument('--config_file', type=str, default=None,
                              help='Path to the configuration file')
    single_parser.add_argument('--output', type=str, default=None,
                              help='Path to save the generated report')
    single_parser.add_argument('--visualize', action='store_true',
                              help='Visualize the result')
    
    # Batch inference
    batch_parser = subparsers.add_parser('batch', help='Batch inference on multiple images')
    batch_parser.add_argument('--image_dir', type=str, required=True,
                             help='Directory containing input images')
    batch_parser.add_argument('--checkpoint', type=str, required=True,
                             help='Path to the model checkpoint')
    batch_parser.add_argument('--config_file', type=str, default=None,
                             help='Path to the configuration file')
    batch_parser.add_argument('--output_dir', type=str, default=None,
                             help='Directory to save the generated reports')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        main(args)
    elif args.mode == 'batch':
        batch_inference(args)
    else:
        parser.print_help() 
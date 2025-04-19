import os
import json
import modal
import torch
from pathlib import Path

# Define Modal volumes to persist data and model checkpoints
data_volume = modal.Volume.from_name("mimic-cxr-data-volume", create_if_missing=True)
model_volume = modal.Volume.from_name("mimic-cxr-model-volume", create_if_missing=True)

# Define the Modal app
app = modal.App("mimic-cxr-radiology")

# Define the Modal image with all required dependencies
image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch==2.2.0",
        "torchvision==0.17.0",
        "transformers==4.39.0",
        "pandas==2.1.4",
        "numpy==1.26.3",
        "matplotlib==3.8.3",
        "scikit-learn==1.4.0",
        "nltk==3.8.1",
        "tqdm==4.66.2",
        "pillow==10.2.0",
        "medspacy>=0.2.0.0",
        "rouge-score==0.1.2",
        "bert-score==0.3.13",
        "tensorboard==2.15.1",
        "opencv-python-headless==4.9.0.80", # Using headless version for Modal
    ])
)

# Define the Modal container with GPU requirements
@app.function(
    image=image,
    volumes={
        "/data": data_volume,
        "/models": model_volume,
    },
    gpu="A10G",  # Using A10G as a cost-effective GPU option
    timeout=60 * 60 * 5  # 5 hour timeout
)
def train_model(stage=1, config_override=None):
    """
    Train the radiology report generation model on Modal.
    
    Args:
        stage (int): Training stage (1 or 2)
        config_override (dict, optional): Override default configuration
    """
    import sys
    # Add the current directory to the Python path
    sys.path.append("/modal")
    
    # Import project modules
    from utils.config import Config
    from training.trainer import train_model as train_fn
    
    print(f"=== Starting Training (Stage {stage}) on Modal ===")
    print(f"Using GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'None'}")
    
    # Create a Config object
    config = Config()
    
    # Update paths for Modal
    config.data_path = "/data"
    config.csv_file = os.path.join("/data", "mimic-cxr-list-filtered.csv")
    config.output_dir = "/models/outputs"
    config.checkpoint_dir = "/models/checkpoints"
    config.log_dir = "/models/logs"
    config.result_dir = "/models/results"
    
    # Override config with any provided values
    if config_override:
        if 'model_config' in config_override:
            config.model_config.update(config_override['model_config'])
        if 'train_config' in config_override:
            config.train_config.update(config_override['train_config'])
        if 'eval_config' in config_override:
            config.eval_config.update(config_override['eval_config'])
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)
    
    # Train the model
    best_val_loss, best_val_metrics = train_fn(config, stage=stage)
    
    # Save final config and metrics
    with open(os.path.join(config.result_dir, f"stage{stage}_config.json"), "w") as f:
        json.dump({
            'model_config': config.model_config,
            'train_config': config.train_config,
            'eval_config': config.eval_config
        }, f, indent=4)
    
    with open(os.path.join(config.result_dir, f"stage{stage}_metrics.json"), "w") as f:
        json.dump({
            'best_val_loss': float(best_val_loss),
            'best_val_metrics': {k: float(v) for k, v in best_val_metrics.items()}
        }, f, indent=4)
    
    print(f"=== Training Completed (Stage {stage}) ===")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation metrics: {best_val_metrics}")
    
    return {
        'best_val_loss': float(best_val_loss),
        'best_val_metrics': {k: float(v) for k, v in best_val_metrics.items()}
    }

@app.function(
    image=image,
    volumes={
        "/data": data_volume,
        "/models": model_volume,
    },
    gpu="A10G",
)
def evaluate_model_modal(stage=1):
    """
    Evaluate the trained model on Modal.
    
    Args:
        stage (int): Stage of the model to evaluate (1 or 2)
    """
    import sys
    sys.path.append("/modal")
    
    from utils.config import Config
    from evaluation.evaluate import evaluate_model as evaluate_fn
    
    print(f"=== Starting Evaluation (Stage {stage}) on Modal ===")
    
    # Create a Config object
    config = Config()
    
    # Update paths for Modal
    config.data_path = "/data"
    config.csv_file = os.path.join("/data", "mimic-cxr-list-filtered.csv")
    config.output_dir = "/models/outputs"
    config.checkpoint_dir = "/models/checkpoints"
    config.log_dir = "/models/logs"
    config.result_dir = "/models/results"
    
    # Use the best model checkpoint for the specified stage
    checkpoint_path = os.path.join(config.checkpoint_dir, f"best_model_stage{stage}.pt")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Evaluate the model
    metrics = evaluate_fn(config, checkpoint_path)
    
    # Save metrics
    with open(os.path.join(config.result_dir, f"eval_stage{stage}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"=== Evaluation Completed ===")
    print(f"Metrics: {metrics}")
    
    return metrics

@app.function(
    image=image,
    volumes={
        "/data": data_volume,
        "/models": model_volume,
    }
)
def upload_data():
    """
    Upload data to the Modal volume. This function should be called before training.
    """
    import shutil
    
    # This path assumes the CSV file is in the local directory from which you're calling the function
    local_csv = "mimic-cxr-list-filtered.csv"
    modal_csv = "/data/mimic-cxr-list-filtered.csv"
    
    # Copy the CSV file to the Modal volume
    if os.path.exists(local_csv):
        shutil.copy(local_csv, modal_csv)
        print(f"Uploaded {local_csv} to Modal volume")
    else:
        print(f"Warning: {local_csv} not found. Please make sure the CSV file exists.")
    
    # Create necessary directories in the data volume
    os.makedirs("/data/images", exist_ok=True)
    os.makedirs("/data/reports", exist_ok=True)
    
    print("Data volume setup complete. Please upload your image and report data to the volume.")
    print("You can use 'modal volume put mimic-cxr-data-volume /path/to/local/images /data/images'")
    print("and 'modal volume put mimic-cxr-data-volume /path/to/local/reports /data/reports'")

@app.function(
    image=image,
    volumes={
        "/models": model_volume,
    }
)
def list_checkpoints():
    """
    List all available model checkpoints in the Modal volume.
    """
    checkpoint_dir = "/models/checkpoints"
    
    if not os.path.exists(checkpoint_dir):
        print("No checkpoints directory found.")
        return []
    
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith(".pt"):
            file_path = os.path.join(checkpoint_dir, filename)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            checkpoints.append({
                "name": filename,
                "path": file_path,
                "size_mb": round(file_size, 2),
                "created": os.path.getctime(file_path)
            })
    
    # Sort by creation time (newest first)
    checkpoints.sort(key=lambda x: x["created"], reverse=True)
    
    for checkpoint in checkpoints:
        print(f"{checkpoint['name']} - {checkpoint['size_mb']} MB")
    
    return checkpoints

# Serve the model for inference
@app.function(
    image=image,
    volumes={
        "/models": model_volume,
    },
    gpu="A10G",
    concurrency_limit=2,
    container_idle_timeout=300,  # 5 minutes
)
def serve_inference():
    """
    Serve the model for inference as an asynchronous web endpoint.
    This creates a persistent endpoint that can receive image inputs and generate reports.
    """
    import sys
    sys.path.append("/modal")
    
    # Import FastAPI for the web service
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import JSONResponse
    import io
    from PIL import Image
    
    # Import project modules
    from utils.config import Config
    from models.report_generator import get_report_generator
    
    web_app = FastAPI(title="Radiology Report Generator")
    
    # Global variables to store the model and config
    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @web_app.on_event("startup")
    async def startup_event():
        nonlocal model
        
        # Create a Config object
        config = Config()
        
        # Update paths for Modal
        config.checkpoint_dir = "/models/checkpoints"
        
        # Load the best model
        checkpoint_path = os.path.join(config.checkpoint_dir, "best_model_stage2.pt")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(config.checkpoint_dir, "best_model_stage1.pt")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No model checkpoint found in {config.checkpoint_dir}")
        
        print(f"Loading model from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get model configuration from checkpoint
        model_config = checkpoint['config']['model_config']
        
        # Create model
        model = get_report_generator(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print("Model loaded and ready for inference")
    
    @web_app.post("/generate_report")
    async def generate_report(file: UploadFile = File(...)):
        if model is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Model not loaded"}
            )
        
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess the image (same as in training)
        from torchvision import transforms
        
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Generate report
        with torch.no_grad():
            generated_ids = model.generate(image_tensor)
            report = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return {"report": report}
    
    return web_app

# Provide a command-line interface for local execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Available commands:")
        print("  upload_data: Upload data to Modal volume")
        print("  train: Train the model (use --stage to specify stage)")
        print("  evaluate: Evaluate the model (use --stage to specify stage)")
        print("  list_checkpoints: List available model checkpoints")
        print("  serve: Serve the model for inference")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "upload_data":
        upload_data.remote()
    
    elif command == "train":
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
        args = parser.parse_args(sys.argv[2:])
        
        result = train_model.remote(stage=args.stage)
        print(f"Training completed with result: {result}")
    
    elif command == "evaluate":
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
        args = parser.parse_args(sys.argv[2:])
        
        result = evaluate_model_modal.remote(stage=args.stage)
        print(f"Evaluation completed with result: {result}")
    
    elif command == "list_checkpoints":
        list_checkpoints.remote()
    
    elif command == "serve":
        import subprocess
        subprocess.run(["modal", "serve", __file__, "--name", "serve_inference"])
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1) 
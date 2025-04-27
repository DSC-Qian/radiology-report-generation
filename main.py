import os
import modal

# 1. Instantiate the App with default image and volumes
image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch", "torchvision",
        "transformers", "huggingface-hub", "tensorboard",
        "evaluate", "bert-score", "rouge-score", "nltk",
        "pandas", "numpy", "tqdm", "Pillow"
    ])
    .add_local_python_source("data")
    .add_local_python_source("utils")
    .add_local_python_source("training")
    .add_local_python_source("evaluation")
    .add_local_python_source("models")
)

data_volume = modal.Volume.from_name("mimic-cxr-data-volume")
model_volume = modal.Volume.from_name("mimic-cxr-model-volume")

app = modal.App(
    "mimic_cxr_report_training",          # Optional name
    image=image,                          # Default image for all functions
    volumes={"/data": data_volume,        # Mount data volume at /data
             "/model": model_volume}      # Mount model volume at /model
)                         

import torch
# Enable TF32 on matmul and cuDNN convs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32    = True
# Enable cuDNN autotuner
torch.backends.cudnn.benchmark   = True

def set_seed(seed: int):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_output_dirs(config):
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)

@app.function(
    gpu="A10G",
    timeout=4 * 60 * 60,
    volumes={"/data": data_volume,        # Mount data volume at /data
             "/model": model_volume}
)
def train_remote(stage: int = 1, checkpoint: str = None, csv_file: str = None):
    from utils.config import Config
    from training.trainer import train_model

    # Override paths for Modal environment
    config = Config()
    if csv_file:
        config.csv_file = csv_file
    config.data_path = "../data/mimic-cxr-jpg"
    config.output_dir = "../model"
    config.checkpoint_dir = os.path.join(config.output_dir, "checkpoints")
    config.log_dir = os.path.join(config.output_dir, "logs")
    config.result_dir = os.path.join(config.output_dir, "results")

    create_output_dirs(config)
    set_seed(config.train_config["seed"])

    # Stage-2 resume logic
    if stage == 2 and checkpoint is None:
        prev = os.path.join(config.checkpoint_dir, "best_model_stage1.pt")
        if os.path.exists(prev):
            checkpoint = prev
        else:
            raise FileNotFoundError(
                "Stage 1 checkpoint not found for stage 2. Provide --checkpoint."
            )

    print(f"=== Training Stage {stage} ===")
    val_loss, val_metrics = train_model(config, stage=stage, resume_from=checkpoint)
    print(f"=== Done Stage {stage}: best val loss {val_loss:.4f} ===")

@app.function(
    gpu="A10G",
    timeout=30 * 60,
    volumes={"/data": data_volume,
             "/model": model_volume}
)
def evaluate_remote(stage: int = 1, checkpoint: str = None, csv_file: str = None):
    from utils.config import Config
    from evaluation.evaluate import evaluate_model

    config = Config()
    if csv_file:
        config.csv_file = csv_file
    config.data_path = "/data"
    config.output_dir = "/model"
    config.checkpoint_dir = os.path.join(config.output_dir, "checkpoints")
    config.log_dir = os.path.join(config.output_dir, "logs")
    config.result_dir = os.path.join(config.output_dir, "results")

    create_output_dirs(config)
    set_seed(config.train_config["seed"])

    if checkpoint is None:
        checkpoint = os.path.join(
            config.checkpoint_dir, f"best_model_stage{stage}.pt"
        )
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"No checkpoint at {checkpoint}.")
    print(f"=== Evaluating Stage {stage} with {checkpoint} ===")
    metrics = evaluate_model(config, checkpoint)
    print("=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

@app.local_entrypoint()
def main(
    mode: str = "train",
    stage: int = 1,
    checkpoint: str = None,
    csv_file: str = None,
):
    """
    CLI entrypoint:
      modal run main.py --mode train --stage 1 [--checkpoint ...] [--csv_file ...]
      modal run main.py --mode eval  --stage 1 [--checkpoint ...] [--csv_file ...]
    """
    if mode == "train":
        train_remote.remote(stage, checkpoint, csv_file)
    elif mode == "eval":
        evaluate_remote.remote(stage, checkpoint, csv_file)
    else:
        raise ValueError("mode must be 'train' or 'eval'")  # :contentReference[oaicite:5]{index=5}

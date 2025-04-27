# Automated Radiology Report Generation from Chest X-Rays

This project implements an end-to-end deep learning system that generates radiology reports from chest X-ray images using the MIMIC-CXR dataset.

## Project Structure

```
.
├── data/
│   └── dataloader.py          # Data loading and preprocessing utilities
├── models/
│   ├── components/
│   │   ├── vision_encoder.py  # Pre-trained chest X-ray model (ResNet/ViT)
│   │   ├── mapping_network.py # Projects image features to language embeddings
│   │   └── language_decoder.py # Text generation model (GPT-2/BioGPT)
│   └── report_generator.py    # Complete end-to-end model
├── training/
│   └── trainer.py             # Training utilities and implementation
├── evaluation/
│   ├── metrics.py             # Implementation of evaluation metrics
│   └── evaluate.py            # Evaluation script
├── utils/
│   ├── config.py              # Configuration parameters
│   └── minimal_train_config.json # Configuration for minimal training setup
├── outputs/
│   ├── checkpoints/           # Saved model checkpoints
│   ├── logs/                  # Training logs
│   └── results/               # Evaluation results
├── images/                    # Chest X-ray images (data subset)
├── reports/                   # Radiology reports (data subset)
├── inference.py               # Script for generating reports from new images
├── main.py                    # Main script to run training and evaluation
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the MIMIC-CXR dataset (or use the provided subset)
4. Organize the data following the MIMIC-CXR structure

## Usage

### Training

```bash
# Stage 1 training (frozen encoder)
python main.py --mode train --stage 1

# Stage 2 training (fine-tuning)
python main.py --mode train --stage 2 --checkpoint path/to/stage1/checkpoint

# Training with minimal configuration for lower resource requirements
python main.py --mode train --stage 1 --config_file utils/minimal_train_config.json --csv_file mimic-cxr-list-minimal.csv
```

### Evaluation

```bash
python main.py --mode eval --checkpoint path/to/checkpoint
```

### Inference

```bash
python inference.py single --image path/to/image.jpg --checkpoint path/to/checkpoint

# Batch inference
python inference.py batch --image_dir path/to/images/ --output_dir path/to/output/ --checkpoint path/to/checkpoint
```

## Model Architecture

- **Vision Encoder**: Pre-trained ResNet-121 (CheXNet) or Vision Transformer (ViT) to extract spatial image features
- **Mapping Network**: Transformer or MLP to project image features into sequence embeddings
- **Language Decoder**: GPT-2 or BioGPT for report generation

## Evaluation Metrics

- **Text Overlap**: BLEU, ROUGE-L, METEOR
- **Clinical Correctness**: CheXpert label F1 or BERTScore

## Dataset Information

The MIMIC-CXR-JPG dataset contains chest X-ray images and associated reports from the Beth Israel Deaconess Medical Center. It includes:

- Over 377,000 images corresponding to 227,835 radiographic studies
- Imaging studies performed at Beth Israel Deaconess Medical Center between 2011-2016
- Free-text radiology reports with findings, impressions, and recommendations

# MIMIC-CXR-JPG Dataset Downloader

This script provides utilities for downloading the MIMIC-CXR-JPG dataset to a Modal volume for use in radiology report generation.

## Prerequisites

1. PhysioNet credentials with access to the MIMIC-CXR-JPG dataset
   - You'll need to complete the required training and get approved for access on the [PhysioNet website](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

2. Modal setup
   - Install Modal: `pip install -r download_requirements.txt`
   - Authenticate with Modal: `modal token new`

## Usage

### Setting up credentials

Set your PhysioNet credentials as environment variables:

```bash
# On Windows
set PHYSIONET_USER=your_username
set PHYSIONET_PASSWORD=your_password

# On Linux/Mac
export PHYSIONET_USER=your_username
export PHYSIONET_PASSWORD=your_password
```

### Running the download

```bash
python download_mimic_cxr.py
```

This will:
1. Create a Modal volume named "mimic-cxr-jpg-volume" if it doesn't exist
2. Download the MIMIC-CXR-JPG dataset to this volume using wget
3. The download is performed on Modal's infrastructure, not your local machine
4. The data will be persisted in the Modal volume for future use
5. Your PhysioNet credentials are automatically passed from your local environment to Modal

### Alternative Download Method

If you have SSH access to PhysioNet, you can modify the main function in the script to use rsync instead:

```python
@app.local_entrypoint()
def main():
    download_mimic_cxr_rsync.remote(
        _env={"PHYSIONET_USER": os.environ.get("PHYSIONET_USER", "")}
    )
```

## Notes

- The dataset is large (several hundred GB), so the download may take considerable time
- The Modal function has a 24-hour timeout, which should be sufficient for most internet connections
- You can resume interrupted downloads thanks to the `-c` (continue) flag used with wget
- The download uses Modal's infrastructure, so you can close your laptop and the download will continue on Modal's servers

## Troubleshooting

- If you encounter authentication issues, verify your PhysioNet credentials are correct and that you have access to the MIMIC-CXR-JPG dataset
- For Modal-related issues, check the Modal documentation at [https://modal.com/docs](https://modal.com/docs)
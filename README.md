# Automated Radiology Report Generation from Chest X-Rays

This project implements an end-to-end deep learning system that generates radiology reports from chest X-ray images using the MIMIC-CXR dataset.

## Project Structure

```
.
├── data/
│   └── dataloader.py          # Data loading and preprocessing utilities
├── models/
│   ├── vision_encoder.py      # Pre-trained chest X-ray model (ResNet/ViT)
│   ├── mapping_network.py     # Projects image features to language embeddings
│   ├── language_decoder.py    # Text generation model (GPT-2/BioGPT)
│   └── report_generator.py    # Complete end-to-end model
├── training/
│   ├── train_stage1.py        # Stage 1 training (frozen encoder)
│   ├── train_stage2.py        # Stage 2 training (fine-tuning)
│   └── trainer.py             # Training utilities
├── evaluation/
│   ├── metrics.py             # Implementation of evaluation metrics
│   └── evaluate.py            # Evaluation script
├── utils/
│   ├── config.py              # Configuration parameters
│   └── visualization.py       # Visualization utilities
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
```

### Evaluation

```bash
python main.py --mode eval --checkpoint path/to/checkpoint
```

### Inference

```bash
python inference.py --image path/to/image.jpg --checkpoint path/to/checkpoint
```

## Model Architecture

- **Vision Encoder**: Pre-trained ResNet-121 (CheXNet) or Vision Transformer (ViT) to extract spatial image features
- **Mapping Network**: Transformer or MLP to project image features into sequence embeddings
- **Language Decoder**: GPT-2 or BioGPT for report generation

## Evaluation Metrics

- **Text Overlap**: BLEU, ROUGE-L, METEOR
- **Clinical Correctness**: CheXpert label F1 or BERTScore
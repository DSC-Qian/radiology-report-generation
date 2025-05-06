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
├── README.md                  # Project documentation
└── preprocessing.py           # Script to preprocess the dataset
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the MIMIC-CXR dataset (or use the provided subset)
4. Organize the data following the MIMIC-CXR structure

## Dataset Information

The MIMIC-CXR dataset contains chest X-ray images and associated reports from the Beth Israel Deaconess Medical Center between 2011-2016. It includes:

- 377,110 images corresponding to 227,835 radiographic studies.
- Each study includes one or more images and a free-text radiology report with findings, impressions, and recommendations.

## Dataset Download

MIMIC-CXR is a restricted-access dataset that requires users to request access. Therefore, we cannot provide a direct download script or similar utility. In the following section, we briefly outline the steps for obtaining access to the dataset and provide guidance on what and how to download once access has been granted.

### Request access
1. Become a credentialed user of [PhysioNet](https://physionet.org/)
2. Complete the required training: CITI Data or Specimens Only Research
3. Sign the data use agreement for [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/) and [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)

### What to download
#### MIMIC-CXR
```
.
├── cxr-record-list.csv.gz     # Image information
├── cxr-study-list.csv.gz      # Report information
└── mimic-cxr-reports.zip      # Radiology reports
```

#### MIMIC-CXR-JPG
```
.
├── files/                               # X-ray images
└── mimic-cxr-2.0.0-metadata.csv.gz      # Image metadata
```

You need to unzip all zip files and rename the image and report folder from "files" to "images" and "reports".

### How to download
#### Method 1
``` 
wget -r -N -c -np --user physionet_username --ask-password given_url 
```
PhysioNet allows users to directly download the dataset from their website. You need to replace ```physionet_username``` and ```given_url``` above to download it. Note that this method can be very slow.

#### Method 2
```
gcloud storage --billing-project project_name cp -r given_url .
```
Both MIMIC-CXR and MIMIC-CXR-JPG have copies saved on Google Cloud Platform. You need to first create a project and link to the dataset, then you can download with the code above by replacing ```project_name``` and ```given_url```. Note that this method is super fast comparing to Method 1, but it is not free. It will cost around $57 credits to download the images.

### Data Preprocessing
You need to run ```python preprocessing.py``` to do the data preprocessing. It will create ```mimic-cxr-list-filtered.csv``` that contains the image and report pairs.

## Model Architecture

- **Vision Encoder**: Pre-trained Vision Transformer (ViT) to extract spatial image features
- **Projection Layer**: A linear layer that maps the output of vision encoder to the input of language decoder
- **Language Decoder**: GPT-2 for report generation

## Evaluation Metrics

- **Text Overlap**: BLEU, ROUGE-L
- **Clinical Correctness**: BERTScore F1, Clinical Terms F1, CheXpert label accuracy

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
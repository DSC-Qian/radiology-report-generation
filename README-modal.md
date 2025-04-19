# Radiology Report Generation on Modal

This repository contains the code for training and deploying a radiology report generation model on [Modal](https://modal.com/), a cloud platform for running machine learning workloads.

## Setup

### Prerequisites

1. Install the Modal CLI:
```bash
pip install modal
```

2. Log in to Modal:
```bash
modal token new
```

3. Clone this repository:
```bash
git clone <repository-url>
cd radiology-report-generation
```

## Data Preparation

Before training, you need to upload your data to Modal volumes. The code expects the following structure:

- A CSV file (`mimic-cxr-list-filtered.csv`) containing image paths and report paths
- Images located in the `images/` directory
- Reports located in the `reports/` directory

To upload your data:

1. First, run the upload command to create the volume and upload the CSV file:
```bash
python modal_app.py upload_data
```

2. Then, upload your image and report directories using the Modal CLI:
```bash
modal volume put mimic-cxr-data-volume /path/to/local/images /data/images
modal volume put mimic-cxr-data-volume /path/to/local/reports /data/reports
```

## Training

The training process is split into two stages:
1. **Stage 1**: Train with frozen encoder
2. **Stage 2**: Fine-tune end-to-end with unfrozen encoder

### Stage 1 Training

```bash
python modal_app.py train --stage 1
```

### Stage 2 Training

After completing Stage 1, run Stage 2 training:

```bash
python modal_app.py train --stage 2
```

## Evaluation

To evaluate the trained model:

```bash
python modal_app.py evaluate --stage 2
```

If you want to evaluate the Stage 1 model, use `--stage 1`.

## Listing Checkpoints

To list all available model checkpoints in your Modal volume:

```bash
python modal_app.py list_checkpoints
```

## Serving the Model for Inference

To deploy the model as a web service for inference:

```bash
python modal_app.py serve
```

This creates an endpoint that accepts image uploads and returns generated reports. The endpoint will be available at a URL provided by Modal.

### Making API Requests

Once the model is deployed, you can make requests to it:

```python
import requests

# Replace with your actual Modal endpoint URL
url = "https://your-modal-endpoint.modal.run/generate_report"

# Load an image
with open("path/to/image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

# Get the generated report
report = response.json()["report"]
print(report)
```

## Advanced Configuration

You can customize the training configuration by modifying the config settings in `utils/config.py`. The Modal implementation uses these settings by default, with paths adapted for the Modal environment.

## GPU Requirements

The Modal implementation uses A10G GPUs by default, which offers a good balance between performance and cost. You can modify the GPU type in `modal_app.py` if you need more or less compute power.

## Troubleshooting

If you encounter issues:

1. Check the Modal logs: `modal app logs mimic-cxr-radiology`
2. Ensure your data is correctly uploaded to the Modal volumes
3. Verify that the CSV file has the correct paths relative to the Modal volume structure

## License

[Add your license information here] 
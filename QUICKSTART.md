# Quick Start Guide

## Setup Instructions

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Download Dataset

Download the PlantVillage dataset from Kaggle:
https://www.kaggle.com/datasets/arjuntejaswi/plant-village

### 3. Prepare Dataset

```bash
cd src
python prepare_data.py --source /path/to/plantvillage --target ../data
```

This will organize the data into:
```
data/
├── train/
│   ├── Healthy/
│   ├── Early_Blight/
│   └── Late_Blight/
├── validation/
└── test/
```

### 4. Explore Data (Optional)

```bash
cd notebooks
jupyter notebook 01_data_exploration.ipynb
```

### 5. Train Models

```bash
cd src
python train.py
```

This will train all three models:
- Custom CNN
- ResNet50 (Transfer Learning)
- EfficientNet-B0 (Transfer Learning)

Training outputs saved to `models/saved_models/`:
- Model checkpoints (.pth)
- Training curves (.png)
- Training history (.json)

### 6. Make Predictions

Single image:
```bash
cd src
python predict.py \
    --image path/to/image.jpg \
    --model ../models/saved_models/resnet50_best.pth \
    --model_type resnet
```

Batch prediction:
```bash
python predict.py \
    --image path/to/images_folder/ \
    --model ../models/saved_models/resnet50_best.pth \
    --model_type resnet
```

## Model Types

- `custom`: Custom CNN (~6M parameters)
- `resnet`: ResNet50 Transfer Learning (~23M parameters)
- `efficientnet`: EfficientNet-B0 Transfer Learning (~4M parameters)

## Expected Results

With proper training on PlantVillage dataset:
- **Custom CNN**: 85-90% accuracy
- **ResNet50**: 95-98% accuracy
- **EfficientNet-B0**: 96-99% accuracy

## Troubleshooting

**CUDA Out of Memory:**
```python
# In train.py, reduce batch size
BATCH_SIZE = 16  # or 8
```

**Slow Training:**
- Ensure GPU is available: `torch.cuda.is_available()`
- Check GPU usage: `nvidia-smi`

**Dataset Not Found:**
```bash
# Verify dataset structure
cd src
python prepare_data.py --verify --target ../data
```

## Next Steps

1. Fine-tune hyperparameters
2. Experiment with different architectures
3. Add ensemble predictions
4. Deploy as web API
5. Create mobile app

## Support

See [README.md](README.md) for detailed documentation.

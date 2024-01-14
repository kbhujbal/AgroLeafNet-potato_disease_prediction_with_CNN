# AgroLeafNet - Potato Disease Classification

Deep learning system for classifying potato leaf diseases using Convolutional Neural Networks (CNN) and Transfer Learning with PyTorch.

## Overview

This project classifies potato leaves into three categories:
- **Healthy**: Normal, disease-free leaves
- **Early Blight**: Caused by *Alternaria solani*
- **Late Blight**: Caused by *Phytophthora infestans*

## Models

The project implements and compares three CNN architectures:

1. **Custom CNN**: Built from scratch with 4 convolutional blocks
2. **ResNet50**: Transfer learning using pre-trained ResNet50
3. **EfficientNet-B0**: Transfer learning using pre-trained EfficientNet-B0

## Project Structure

```
AgroLeafNet/
├── data/
│   ├── train/
│   │   ├── Healthy/
│   │   ├── Early_Blight/
│   │   └── Late_Blight/
│   ├── validation/
│   │   ├── Healthy/
│   │   ├── Early_Blight/
│   │   └── Late_Blight/
│   └── test/
│       ├── Healthy/
│       ├── Early_Blight/
│       └── Late_Blight/
├── models/
│   └── saved_models/
├── notebooks/
├── src/
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── models.py            # Model architectures
│   ├── train.py             # Training script
│   └── predict.py           # Inference script
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd AgroLeafNet
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

### Using PlantVillage Dataset

1. Download the PlantVillage dataset from [Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)

2. Extract and organize the potato images:
   - Filter only potato-related classes: `Potato___Early_blight`, `Potato___Late_blight`, `Potato___healthy`

3. Split the data into train/validation/test sets (recommended ratio: 70/15/15):

```python
# Example script to organize data
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Set paths
source_dir = Path('path/to/plantvillage')
target_dir = Path('data')

# Class mapping
classes = {
    'Potato___healthy': 'Healthy',
    'Potato___Early_blight': 'Early_Blight',
    'Potato___Late_blight': 'Late_Blight'
}

# Create directory structure
for split in ['train', 'validation', 'test']:
    for class_name in classes.values():
        (target_dir / split / class_name).mkdir(parents=True, exist_ok=True)

# Copy and split images
# ... (implement your data splitting logic)
```

### Data Structure

Ensure your data directory follows this structure:
```
data/
├── train/
│   ├── Healthy/
│   ├── Early_Blight/
│   └── Late_Blight/
├── validation/
│   ├── Healthy/
│   ├── Early_Blight/
│   └── Late_Blight/
└── test/
    ├── Healthy/
    ├── Early_Blight/
    └── Late_Blight/
```

## Training

### Train All Models

To train all three models (Custom CNN, ResNet50, EfficientNet-B0):

```bash
cd src
python train.py
```

### Training Configuration

You can modify training parameters in [train.py](src/train.py):

```python
DATA_DIR = 'data'
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
```

### Training Features

- Data augmentation (rotation, flip, color jitter)
- Learning rate scheduling (ReduceLROnPlateau)
- Model checkpointing (saves best model)
- Training history logging
- Automatic visualization of training curves

### Output

Training generates the following outputs in `models/saved_models/`:

- `{model_name}_best.pth`: Best model checkpoint
- `{model_name}_final.pth`: Final model checkpoint
- `{model_name}_history.json`: Training history
- `{model_name}_training_curves.png`: Loss and accuracy plots
- `classification_report.txt`: Detailed metrics
- `confusion_matrix.png`: Confusion matrix visualization

## Inference

### Single Image Prediction

```bash
cd src
python predict.py \
    --image path/to/potato_leaf.jpg \
    --model ../models/saved_models/resnet50_best.pth \
    --model_type resnet \
    --device cuda
```

### Batch Prediction

```bash
cd src
python predict.py \
    --image path/to/images_folder/ \
    --model ../models/saved_models/resnet50_best.pth \
    --model_type resnet \
    --device cuda
```

### Model Types

- `custom`: Custom CNN
- `resnet`: ResNet50 Transfer Learning
- `efficientnet`: EfficientNet-B0 Transfer Learning

### Example Output

```
==============================================================
Image: path/to/potato_leaf.jpg
--------------------------------------------------------------
Predicted Class: Early_Blight
Confidence: 96.73%

All Probabilities:
  Healthy: 1.23%
  Early_Blight: 96.73%
  Late_Blight: 2.04%
==============================================================
```

## Model Architectures

### Custom CNN

- 4 Convolutional blocks (32, 64, 128, 256 filters)
- Batch Normalization after each conv layer
- MaxPooling for downsampling
- 3 Fully connected layers with dropout
- ~6M parameters

### ResNet50 (Transfer Learning)

- Pre-trained on ImageNet
- Early layers frozen for transfer learning
- Custom classification head
- ~23M parameters (10M trainable)

### EfficientNet-B0 (Transfer Learning)

- Pre-trained on ImageNet
- Efficient architecture with compound scaling
- Custom classification head
- ~4M parameters (3M trainable)

## Data Augmentation

Applied during training:
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.3)
- Random rotation (±20°)
- Color jitter (brightness, contrast, saturation)
- Random affine transformation
- Normalization (ImageNet statistics)

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Per-class metrics

## Performance Tips

### For Better Accuracy:
- Use more training data
- Increase epochs (30-50)
- Fine-tune learning rate
- Experiment with different augmentations
- Use ensemble predictions

### For Faster Training:
- Use GPU (CUDA)
- Reduce batch size if out of memory
- Use mixed precision training
- Start with transfer learning models

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use smaller image size
- Use gradient accumulation

### Low Accuracy
- Check data quality and labels
- Increase training epochs
- Add more data augmentation
- Try different learning rates

### Slow Training
- Enable CUDA if available
- Increase batch size (if memory allows)
- Reduce number of workers in data loader

## Future Enhancements

- [ ] Web interface for predictions
- [ ] Mobile app integration
- [ ] Real-time disease detection
- [ ] Multi-crop support
- [ ] Severity assessment
- [ ] Treatment recommendations
- [ ] Model quantization for edge deployment

## Dataset Citation

If using PlantVillage dataset:

```
@article{hughes2015open,
  title={An open access repository of images on plant health to enable the development of mobile disease diagnostics},
  author={Hughes, David P and Salath{\'e}, Marcel},
  journal={arXiv preprint arXiv:1511.08060},
  year={2015}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- PlantVillage Dataset
- PyTorch Team
- Transfer Learning research community

## Contact

For questions or issues, please open an issue on GitHub.

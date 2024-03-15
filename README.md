# AgroLeafNet

A deep learning project to classify potato leaf diseases using CNNs and transfer learning.

## What it does

Identifies three conditions in potato leaves:
- **Healthy** - disease-free leaves
- **Early Blight** - fungal infection (Alternaria solani)
- **Late Blight** - the disease that caused the Irish potato famine (Phytophthora infestans)

## Models

I've implemented three approaches to compare performance:

1. **Custom CNN** - built from scratch (4 conv blocks, ~6M params)
2. **ResNet50** - transfer learning from ImageNet (~23M params, 10M trainable)
3. **EfficientNet-B0** - more efficient transfer learning (~4M params, 3M trainable)

## Quick Start

### Installation

```bash
git clone <your-repo>
cd AgroLeafNet
pip install -r requirements.txt
```

### Get the Data

Download PlantVillage dataset from [Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village), then organize it:

```bash
cd src
python prepare_data.py --source /path/to/plantvillage --target ../data
```

This splits it into train/val/test (70/15/15).

### Train

```bash
cd src
python train.py
```

Training takes about 1-1.5 hours for all three models (with GPU/MPS). Models are saved to `models/saved_models/`.

### Predict

Single image:
```bash
python predict.py --image leaf.jpg --model ../models/saved_models/resnet50_best.pth --model_type resnet
```

Batch prediction:
```bash
python predict.py --image /folder/of/images --model ../models/saved_models/resnet50_best.pth --model_type resnet
```

## Project Structure

```
src/
├── data_loader.py    # handles data loading & augmentation
├── models.py         # model architectures
├── train.py          # training loop
└── predict.py        # inference

data/
├── train/
├── validation/
└── test/

models/saved_models/  # trained models go here
notebooks/            # exploratory analysis
```

## What happens during training

- Data augmentation (flips, rotations, color jitter)
- Automatic learning rate scheduling
- Saves best model based on validation accuracy
- Generates training curves and confusion matrix
- Full classification report with precision/recall/F1

## Results

Expected accuracy on PlantVillage:
- Custom CNN: 85-90%
- ResNet50: 95-98%
- EfficientNet-B0: 96-99%

The imbalanced dataset (152 healthy vs 1000 disease images) makes the healthy class harder to predict.

## Troubleshooting

**Out of memory?** Lower the batch size in `train.py` (try 16 or 8).

**Slow training?** Make sure you're using GPU/CUDA or MPS (Apple Silicon). Check with `torch.cuda.is_available()` or `torch.backends.mps.is_available()`.

**Import errors?** You might be using a different Python environment than where you installed packages. Check with `which python`.

## Dataset

Using the PlantVillage dataset. If you use it in research, cite:

```
Hughes, D.P. and Salathe, M., 2015.
An open access repository of images on plant health to enable the development of mobile disease diagnostics.
arXiv preprint arXiv:1511.08060.
```

## License

MIT

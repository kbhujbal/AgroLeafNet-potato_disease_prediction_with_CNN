"""
Quick test to verify setup before training
"""

import torch
from models import get_model
from data_loader import PotatoDataset

print("=" * 60)
print("AgroLeafNet Setup Test")
print("=" * 60)

# Check device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"\n✓ Device: {device}")

# Test data loading
print("\nTesting data loading...")
dataset = PotatoDataset('../data', image_size=224, batch_size=4)
train_loader, val_loader, test_loader, class_names = dataset.get_data_loaders()

print(f"✓ Training samples: {len(train_loader.dataset)}")
print(f"✓ Validation samples: {len(val_loader.dataset)}")
print(f"✓ Test samples: {len(test_loader.dataset)}")
print(f"✓ Classes: {class_names}")

# Test models
print("\nTesting models...")
for model_name in ['custom', 'resnet', 'efficientnet']:
    model = get_model(model_name, num_classes=len(class_names), pretrained=False)
    model.to(device)
    print(f"✓ {model_name.capitalize()} model initialized")

# Test forward pass
print("\nTesting forward pass...")
batch = next(iter(train_loader))
images, labels = batch
images = images.to(device)
output = model(images)
print(f"✓ Input shape: {images.shape}")
print(f"✓ Output shape: {output.shape}")

print("\n" + "=" * 60)
print("All tests passed! Ready to train.")
print("=" * 60)
print("\nRun: python train.py")

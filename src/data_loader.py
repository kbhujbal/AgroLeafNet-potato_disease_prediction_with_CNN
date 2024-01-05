"""
Data loading and preprocessing utilities for potato disease classification
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from pathlib import Path
import os


class PotatoDataset:
    """
    Dataset handler for potato disease classification
    Classes: Healthy, Early Blight, Late Blight
    """

    def __init__(self, data_dir, image_size=224, batch_size=32):
        """
        Args:
            data_dir: Root directory containing train/validation/test folders
            image_size: Size to resize images to (default: 224 for transfer learning)
            batch_size: Batch size for data loaders
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size

        # Define data transformations
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.val_test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def get_data_loaders(self):
        """
        Create train, validation, and test data loaders

        Returns:
            train_loader, val_loader, test_loader, class_names
        """
        train_dir = self.data_dir / 'train'
        val_dir = self.data_dir / 'validation'
        test_dir = self.data_dir / 'test'

        # Create datasets
        train_dataset = datasets.ImageFolder(
            root=train_dir,
            transform=self.train_transform
        )

        val_dataset = datasets.ImageFolder(
            root=val_dir,
            transform=self.val_test_transform
        )

        test_dataset = datasets.ImageFolder(
            root=test_dir,
            transform=self.val_test_transform
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        class_names = train_dataset.classes

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Classes: {class_names}")

        return train_loader, val_loader, test_loader, class_names


def get_transforms(image_size=224, augment=True):
    """
    Get image transformations

    Args:
        image_size: Target image size
        augment: Whether to apply data augmentation

    Returns:
        transform pipeline
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

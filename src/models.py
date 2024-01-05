"""
CNN Models for Potato Disease Classification
- Custom CNN: Built from scratch
- Transfer Learning: ResNet50 and EfficientNet-B0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CustomCNN(nn.Module):
    """
    Custom CNN architecture for potato disease classification

    Architecture:
    - 4 Convolutional blocks with BatchNorm and MaxPooling
    - Dropout for regularization
    - 2 Fully connected layers
    - Output: 3 classes (Healthy, Early Blight, Late Blight)
    """

    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(CustomCNN, self).__init__()

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Convolutional Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Fully Connected Layers
        # After 4 pooling layers: 224 -> 112 -> 56 -> 28 -> 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Block 4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


class ResNetTransferLearning(nn.Module):
    """
    Transfer Learning model using ResNet50
    Pre-trained on ImageNet, fine-tuned for potato disease classification
    """

    def __init__(self, num_classes=3, pretrained=True, freeze_layers=True):
        """
        Args:
            num_classes: Number of output classes (default: 3)
            pretrained: Use ImageNet pre-trained weights
            freeze_layers: Freeze early layers for transfer learning
        """
        super(ResNetTransferLearning, self).__init__()

        # Load pre-trained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)

        # Freeze early layers if specified
        if freeze_layers:
            for param in list(self.resnet.parameters())[:-10]:
                param.requires_grad = False

        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


class EfficientNetTransferLearning(nn.Module):
    """
    Transfer Learning model using EfficientNet-B0
    More efficient architecture with better accuracy/parameter ratio
    """

    def __init__(self, num_classes=3, pretrained=True, freeze_layers=True):
        """
        Args:
            num_classes: Number of output classes (default: 3)
            pretrained: Use ImageNet pre-trained weights
            freeze_layers: Freeze early layers for transfer learning
        """
        super(EfficientNetTransferLearning, self).__init__()

        # Load pre-trained EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(pretrained=pretrained)

        # Freeze early layers if specified
        if freeze_layers:
            for param in list(self.efficientnet.parameters())[:-20]:
                param.requires_grad = False

        # Replace classifier
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)


def get_model(model_name='resnet', num_classes=3, pretrained=True):
    """
    Factory function to create models

    Args:
        model_name: 'custom', 'resnet', or 'efficientnet'
        num_classes: Number of output classes
        pretrained: Use pre-trained weights (for transfer learning models)

    Returns:
        model instance
    """
    if model_name.lower() == 'custom':
        return CustomCNN(num_classes=num_classes)
    elif model_name.lower() == 'resnet':
        return ResNetTransferLearning(num_classes=num_classes, pretrained=pretrained)
    elif model_name.lower() == 'efficientnet':
        return EfficientNetTransferLearning(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from 'custom', 'resnet', 'efficientnet'")


if __name__ == "__main__":
    # Test models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Testing Custom CNN...")
    custom_model = CustomCNN()
    print(f"Parameters: {sum(p.numel() for p in custom_model.parameters()):,}")

    print("\nTesting ResNet Transfer Learning...")
    resnet_model = ResNetTransferLearning()
    print(f"Parameters: {sum(p.numel() for p in resnet_model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in resnet_model.parameters() if p.requires_grad):,}")

    print("\nTesting EfficientNet Transfer Learning...")
    efficientnet_model = EfficientNetTransferLearning()
    print(f"Parameters: {sum(p.numel() for p in efficientnet_model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in efficientnet_model.parameters() if p.requires_grad):,}")

    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = custom_model(dummy_input)
    print(f"\nOutput shape: {output.shape}")

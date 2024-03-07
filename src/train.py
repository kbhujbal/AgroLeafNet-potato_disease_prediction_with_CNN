"""
Training script for potato disease classification models
Includes training loop, validation, and model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from pathlib import Path
import time
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models import get_model
from data_loader import PotatoDataset


class Trainer:
    """
    Training manager for potato disease classification models
    """

    def __init__(self, model, train_loader, val_loader, class_names,
                 device='cuda', learning_rate=0.001, model_name='model'):
        """
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            class_names: List of class names
            device: 'cuda' or 'cpu'
            learning_rate: Initial learning rate
            model_name: Name for saving model
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names
        self.device = device
        self.model_name = model_name

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                          patience=3, verbose=True)

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        self.best_val_acc = 0.0

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss / (pbar.n + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({
                    'loss': f'{running_loss / (pbar.n + 1):.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def train(self, epochs=25, save_dir='models/saved_models'):
        """
        Train the model for specified epochs

        Args:
            epochs: Number of epochs
            save_dir: Directory to save model checkpoints
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nTraining {self.model_name} for {epochs} epochs...")
        print(f"Device: {self.device}")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 60)

            # Train
            train_loss, train_acc = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validate
            val_loss, val_acc = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(save_dir / f'{self.model_name}_best.pth')
                print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")

        # Training complete
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
        print(f"Best Val Accuracy: {self.best_val_acc:.2f}%")

        # Save final model and history
        self.save_model(save_dir / f'{self.model_name}_final.pth')
        self.save_history(save_dir / f'{self.model_name}_history.json')
        self.plot_training_history(save_dir / f'{self.model_name}_training_curves.png')

    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'class_names': self.class_names
        }, path)

    def save_history(self, path):
        """Save training history as JSON"""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=4)

    def plot_training_history(self, save_path):
        """Plot and save training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {save_path}")


def evaluate_model(model, test_loader, class_names, device='cuda', save_dir='models/saved_models'):
    """
    Evaluate model on test set and generate metrics

    Args:
        model: Trained model
        test_loader: Test data loader
        class_names: List of class names
        device: Device to run on
        save_dir: Directory to save evaluation results
    """
    model.eval()
    all_preds = []
    all_labels = []

    print("\nEvaluating model on test set...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Classification report
    print("\nClassification Report:")
    print("=" * 60)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)

    # Save report
    save_dir = Path(save_dir)
    with open(save_dir / 'classification_report.txt', 'w') as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, save_dir / 'confusion_matrix.png')

    # Calculate accuracy
    accuracy = 100. * (all_preds == all_labels).sum() / len(all_labels)
    print(f"\nTest Accuracy: {accuracy:.2f}%")

    return accuracy, report, cm


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


if __name__ == "__main__":
    # Configuration
    DATA_DIR = '../data'  # Relative to src directory
    BATCH_SIZE = 32
    EPOCHS = 25
    LEARNING_RATE = 0.001
    IMAGE_SIZE = 224

    # Set device (supports CUDA, MPS for Apple Silicon, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
    dataset = PotatoDataset(DATA_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
    train_loader, val_loader, test_loader, class_names = dataset.get_data_loaders()

    # Train Custom CNN
    print("\n" + "=" * 60)
    print("Training Custom CNN")
    print("=" * 60)
    custom_model = get_model('custom', num_classes=len(class_names), pretrained=False)
    custom_trainer = Trainer(custom_model, train_loader, val_loader, class_names,
                             device=device, learning_rate=LEARNING_RATE, model_name='custom_cnn')
    custom_trainer.train(epochs=EPOCHS)

    # Train ResNet (Transfer Learning)
    print("\n" + "=" * 60)
    print("Training ResNet50 (Transfer Learning)")
    print("=" * 60)
    resnet_model = get_model('resnet', num_classes=len(class_names), pretrained=True)
    resnet_trainer = Trainer(resnet_model, train_loader, val_loader, class_names,
                             device=device, learning_rate=LEARNING_RATE, model_name='resnet50')
    resnet_trainer.train(epochs=EPOCHS)

    # Train EfficientNet (Transfer Learning)
    print("\n" + "=" * 60)
    print("Training EfficientNet-B0 (Transfer Learning)")
    print("=" * 60)
    efficientnet_model = get_model('efficientnet', num_classes=len(class_names), pretrained=True)
    efficientnet_trainer = Trainer(efficientnet_model, train_loader, val_loader, class_names,
                                   device=device, learning_rate=LEARNING_RATE, model_name='efficientnet_b0')
    efficientnet_trainer.train(epochs=EPOCHS)

    print("\n" + "=" * 60)
    print("All models trained successfully!")
    print("=" * 60)

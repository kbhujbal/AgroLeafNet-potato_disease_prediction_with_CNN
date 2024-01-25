"""
Prediction/Inference script for potato disease classification
Load trained models and make predictions on new images
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import argparse

from models import get_model


class PotatoDiseasePredictor:
    """
    Predictor for potato disease classification
    """

    def __init__(self, model_path, model_type='resnet', device='cuda'):
        """
        Args:
            model_path: Path to saved model checkpoint
            model_type: Type of model ('custom', 'resnet', 'efficientnet')
            device: Device to run on ('cuda', 'mps', or 'cpu')
        """
        # Auto-detect best available device
        if device == 'cuda' and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = torch.device(device)
        self.model_type = model_type

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_names = checkpoint.get('class_names', ['Early_Blight', 'Late_Blight', 'Healthy'])

        # Initialize model
        self.model = get_model(model_type, num_classes=len(self.class_names), pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        print(f"Model loaded: {model_type}")
        print(f"Classes: {self.class_names}")
        print(f"Device: {self.device}")

    def predict_image(self, image_path):
        """
        Predict disease for a single image

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class = self.class_names[predicted_idx.item()]
        confidence_score = confidence.item() * 100

        # Get probabilities for all classes
        all_probs = {
            self.class_names[i]: probabilities[0][i].item() * 100
            for i in range(len(self.class_names))
        }

        result = {
            'predicted_class': predicted_class,
            'confidence': confidence_score,
            'probabilities': all_probs
        }

        return result

    def predict_batch(self, image_paths):
        """
        Predict diseases for multiple images

        Args:
            image_paths: List of image paths

        Returns:
            List of prediction results
        """
        results = []
        for image_path in image_paths:
            result = self.predict_image(image_path)
            result['image_path'] = str(image_path)
            results.append(result)

        return results

    def print_prediction(self, result):
        """Pretty print prediction result"""
        print("\n" + "=" * 60)
        print(f"Image: {result.get('image_path', 'N/A')}")
        print("-" * 60)
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print("\nAll Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob:.2f}%")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Potato Disease Prediction')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image file or directory')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='resnet',
                       choices=['custom', 'resnet', 'efficientnet'],
                       help='Type of model')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run on')

    args = parser.parse_args()

    # Initialize predictor
    predictor = PotatoDiseasePredictor(
        model_path=args.model,
        model_type=args.model_type,
        device=args.device
    )

    # Check if input is file or directory
    input_path = Path(args.image)

    if input_path.is_file():
        # Single image prediction
        result = predictor.predict_image(input_path)
        result['image_path'] = str(input_path)
        predictor.print_prediction(result)

    elif input_path.is_dir():
        # Batch prediction
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = [
            p for p in input_path.iterdir()
            if p.suffix.lower() in image_extensions
        ]

        print(f"\nFound {len(image_paths)} images")
        results = predictor.predict_batch(image_paths)

        for result in results:
            predictor.print_prediction(result)

    else:
        print(f"Error: {input_path} is not a valid file or directory")


if __name__ == "__main__":
    # Example usage
    print("Potato Disease Predictor")
    print("=" * 60)
    print("\nUsage:")
    print("  Single image:")
    print("    python predict.py --image path/to/image.jpg --model models/saved_models/resnet50_best.pth --model_type resnet")
    print("\n  Batch prediction:")
    print("    python predict.py --image path/to/images/ --model models/saved_models/resnet50_best.pth --model_type resnet")
    print("\n" + "=" * 60)

    # Run main if arguments provided
    import sys
    if len(sys.argv) > 1:
        main()

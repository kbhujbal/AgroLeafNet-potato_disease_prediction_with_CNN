"""
Helper script to organize PlantVillage dataset into train/val/test splits
"""

import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
import argparse


def organize_dataset(source_dir, target_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Organize PlantVillage potato images into train/val/test splits

    Args:
        source_dir: Path to PlantVillage dataset
        target_dir: Path to output directory (data/)
        train_ratio: Ratio of training data (default: 0.7)
        val_ratio: Ratio of validation data (default: 0.15)
        test_ratio: Ratio of test data (default: 0.15)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    # Class mapping from PlantVillage to our format
    class_mapping = {
        'Potato___healthy': 'Healthy',
        'Potato___Early_blight': 'Early_Blight',
        'Potato___Late_blight': 'Late_Blight'
    }

    print("=" * 60)
    print("Organizing Potato Disease Dataset")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Split ratio - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    print()

    # Create directory structure
    for split in ['train', 'validation', 'test']:
        for class_name in class_mapping.values():
            (target_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    # Process each class
    for source_class, target_class in class_mapping.items():
        source_class_dir = source_dir / source_class

        if not source_class_dir.exists():
            print(f"Warning: {source_class_dir} not found. Skipping...")
            continue

        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        images = [
            f for f in source_class_dir.iterdir()
            if f.suffix in image_extensions
        ]

        print(f"\nProcessing {source_class} -> {target_class}")
        print(f"Total images: {len(images)}")

        if len(images) == 0:
            print(f"No images found in {source_class_dir}")
            continue

        # Shuffle images
        random.shuffle(images)

        # Calculate split sizes
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        # Split images
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]

        print(f"  Train: {len(train_images)}")
        print(f"  Validation: {len(val_images)}")
        print(f"  Test: {len(test_images)}")

        # Copy files to respective directories
        for img in tqdm(train_images, desc=f"  Copying train"):
            dst = target_dir / 'train' / target_class / img.name
            shutil.copy2(img, dst)

        for img in tqdm(val_images, desc=f"  Copying val"):
            dst = target_dir / 'validation' / target_class / img.name
            shutil.copy2(img, dst)

        for img in tqdm(test_images, desc=f"  Copying test"):
            dst = target_dir / 'test' / target_class / img.name
            shutil.copy2(img, dst)

    print("\n" + "=" * 60)
    print("Dataset organization complete!")
    print("=" * 60)

    # Print summary
    print("\nDataset Summary:")
    for split in ['train', 'validation', 'test']:
        print(f"\n{split.capitalize()}:")
        for class_name in class_mapping.values():
            class_dir = target_dir / split / class_name
            n_images = len(list(class_dir.glob('*')))
            print(f"  {class_name}: {n_images} images")


def verify_dataset(data_dir):
    """
    Verify dataset structure and print statistics

    Args:
        data_dir: Path to data directory
    """
    data_dir = Path(data_dir)

    print("\n" + "=" * 60)
    print("Dataset Verification")
    print("=" * 60)

    class_names = ['Healthy', 'Early_Blight', 'Late_Blight']
    splits = ['train', 'validation', 'test']

    for split in splits:
        print(f"\n{split.capitalize()}:")
        split_dir = data_dir / split

        if not split_dir.exists():
            print(f"  ❌ Directory not found: {split_dir}")
            continue

        for class_name in class_names:
            class_dir = split_dir / class_name

            if not class_dir.exists():
                print(f"  ❌ {class_name}: Directory not found")
                continue

            images = list(class_dir.glob('*'))
            print(f"  ✓ {class_name}: {len(images)} images")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare PlantVillage potato dataset')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to PlantVillage dataset directory')
    parser.add_argument('--target', type=str, default='../data',
                       help='Path to output data directory (default: ../data)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training data ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation data ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test data ratio (default: 0.15)')
    parser.add_argument('--verify', action='store_true',
                       help='Only verify existing dataset structure')

    args = parser.parse_args()

    if args.verify:
        verify_dataset(args.target)
    else:
        # Check ratios sum to 1
        total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            print(f"Error: Ratios must sum to 1.0 (current sum: {total_ratio})")
            exit(1)

        organize_dataset(
            source_dir=args.source,
            target_dir=args.target,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )

        # Verify after organization
        verify_dataset(args.target)

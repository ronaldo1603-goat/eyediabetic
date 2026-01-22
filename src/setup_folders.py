"""
Helper script to create required folder structure for training
Run this before training to ensure all folders exist
"""

import os

def create_folders():
    """Create required folder structure for training"""
    
    folders = [
        'inputs',
        'inputs/train_images',
        'inputs/test_images',
        'models',
        'probs',
        'submissions'  # For test.py output
    ]
    
    print("Creating folder structure...")
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"  ✓ {folder}/")
    
    print("\nFolder structure created successfully!")
    print("\nNote: Images will be preprocessed on-the-fly during training/testing.")
    print("No need to preprocess and save to processed/ folder.")
    print("\nNext steps:")
    print("1. Place train.csv in inputs/train.csv")
    print("2. Place training images in inputs/train_images/")
    print("3. (Optional) Place test.csv in inputs/test.csv")
    print("4. (Optional) Place test images in inputs/test_images/")
    print("\nThen run training:")
    print("  python train.py --arch se_resnext50_32x4d --train_dataset aptos2019")

if __name__ == "__main__":
    create_folders()


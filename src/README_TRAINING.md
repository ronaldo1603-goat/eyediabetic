# Training Guide

## Prerequisites

### Python Version
- **Python 3.11** is required for this project

### Install Dependencies

Install all required packages using pip:

```bash
cd src
pip install -r requirements.txt
```

Or install individually:
```bash
pip install torch torchvision pretrainedmodels efficientnet-pytorch
pip install opencv-python Pillow scikit-image
pip install numpy pandas scipy scikit-learn joblib matplotlib tqdm
```

**Key Dependencies:**
- `torch>=1.7.0` - PyTorch deep learning framework
- `torchvision>=0.8.0` - Computer vision utilities
- `pretrainedmodels>=0.7.4` - Pretrained model architectures (SE-ResNeXt, SENet, etc.)
- `efficientnet-pytorch>=0.7.0` - EfficientNet models
- `opencv-python>=4.5.0` - Image processing
- `scikit-image>=0.17.0` - Image analysis
- Other scientific computing libraries (numpy, pandas, scipy, scikit-learn)

## Download Training Data

### Option 1: Using Kaggle API (Recommended)

1. **Install Kaggle API:**
   ```bash
   pip install kaggle
   ```

2. **Setup Kaggle credentials:**
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token" to download `kaggle.json`
   - Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac)

3. **Download competition data:**
   ```bash
   # Navigate to project root (not src folder)
   cd ..
   kaggle competitions download -c aptos2019-blindness-detection
   ```

4. **Extract and organize data:**
   ```bash
   # Extract zip file
   unzip aptos2019-blindness-detection.zip
   
   # Create inputs folder structure
   cd eyediabetic_predict/src
   python setup_folders.py
   
   # Move files to correct locations
   cd ../..
   mv aptos2019-blindness-detection/train.csv eyediabetic_predict/src/inputs/
   mv aptos2019-blindness-detection/train_images eyediabetic_predict/src/inputs/
   mv aptos2019-blindness-detection/test.csv eyediabetic_predict/src/inputs/
   mv aptos2019-blindness-detection/test_images eyediabetic_predict/src/inputs/
   ```

### Option 2: Manual Download from Kaggle

1. **Visit the competition page:**
   - Go to: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
   - You need to accept the competition rules first (if not already done)

2. **Download files:**
   - Click "Download All" or download individual files:
     - `train.csv` - Training labels
     - `train_images.zip` - Training images (extract to get folder)
     - `test.csv` - Test labels (optional)
     - `test_images.zip` - Test images (optional, extract to get folder)

3. **Extract and organize:**
   ```bash
   # Extract zip files
   unzip train_images.zip
   unzip test_images.zip  # if downloaded
   
   # Create folder structure
   cd eyediabetic_predict/src
   python setup_folders.py
   
   # Move files to inputs/ folder
   # (adjust paths based on where you extracted files)
   cp train.csv inputs/
   cp -r train_images inputs/
   cp test.csv inputs/  # if downloaded
   cp -r test_images inputs/  # if downloaded
   ```

### Verify Data Structure

After downloading, your `src/inputs/` folder should look like:
```
inputs/
├── train.csv              # 3,662 rows with id_code and diagnosis
├── train_images/          # 3,662 PNG images
│   ├── 000c1434d8d7.png
│   ├── 002c21358ce6.png
│   └── ...
├── test.csv               # (optional) Test labels
└── test_images/           # (optional) Test images
    └── ...
```

## Folder Structure Required

The training code expects the following folder structure (Kaggle-style):

```
project_root/
├── inputs/
│   ├── train.csv                    # APTOS 2019 training labels
│   ├── train_images/                # APTOS 2019 training images
│   │   └── *.png
│   ├── test.csv                     # APTOS 2019 test labels (optional)
│   ├── test_images/                 # APTOS 2019 test images (optional)
│   │   └── *.png
│   ├── diabetic-retinopathy-resized/  # Optional: additional dataset
│   │   ├── trainLabels.csv
│   │   └── resized_train/
│   └── strMd5.csv                   # Optional: for duplicate removal
├── models/                          # Auto-generated: saved models
│   └── <model_name>/
│       ├── model_1.pth
│       ├── model_2.pth
│       ├── ...
│       ├── log_1.csv
│       ├── args.pkl
│       └── args.txt
└── probs/                           # Auto-generated: prediction results
    └── <model_name>.csv
```

## Quick Setup

### Step 1: Install Python 3.11 and Dependencies

```bash
# Verify Python version
python --version  # Should be 3.11.x

# Install dependencies
cd src
pip install -r requirements.txt
```

### Step 2: Download Data

Follow the "Download Training Data" section above to get the APTOS 2019 dataset from Kaggle.

### Step 3: Create Folder Structure

```bash
cd src
python setup_folders.py
```

Or manually:
```bash
mkdir -p inputs/train_images inputs/test_images models probs submissions
```

### Step 4: Verify Data

Make sure your data is in the correct location:
- `inputs/train.csv` - Training labels
- `inputs/train_images/` - Training images (*.png)
- (Optional) `inputs/test.csv` and `inputs/test_images/` for testing

### Step 5: Run Training

```bash
cd src
python train.py --arch se_resnext50_32x4d --train_dataset aptos2019
```

## Training Command Examples

### Basic training (APTOS 2019 only):
```bash
python train.py --arch se_resnext50_32x4d --train_dataset aptos2019 --epochs 30
```

### With custom name:
```bash
python train.py --arch se_resnext50_32x4d --name my_model --train_dataset aptos2019
```

### With pseudo labels:
```bash
python train.py --arch se_resnext50_32x4d --train_dataset aptos2019 --pseudo_labels previous_model_name
```

## Important Notes

- **Working Directory**: Run training from `src/` folder
- **Paths**: All paths are relative to where you run the script
- **Preprocessing**: Images are preprocessed **on-the-fly** during training/testing (no need for `processed/` folder)
- **Models**: Saved models go to `models/<model_name>/` folder
- **Logs**: Training logs are saved as CSV files in `models/<model_name>/`

## Configuration

The code uses hardcoded paths matching Kaggle competition structure. This is intentional to:
- Match the original solution exactly
- Make it easy to compare with original code
- Maintain compatibility with existing trained models

**Important Changes:**
- Images are now loaded directly from `inputs/train_images/` and `inputs/test_images/`
- Preprocessing (scale_radius) is performed **on-the-fly** in the Dataset class
- No need for `processed/` folder anymore - saves disk space and simplifies workflow

## Troubleshooting

**Error: File not found 'inputs/train.csv'**
- Make sure you're running from `src/` folder
- Create `inputs/` folder and place your data there

**Error: No images found**
- Check that images are in `inputs/train_images/` folder
- Verify image filenames match the IDs in `train.csv`

**CUDA out of memory**
- Reduce batch size: `--batch_size 16`
- Use smaller model or reduce image size


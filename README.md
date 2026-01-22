# EyeDiabetic Predict - Diabetic Retinopathy Detection

This project provides a complete solution for diabetic retinopathy detection using deep learning, based on the APTOS 2019 Blindness Detection competition.

## Project Structure

```
eyediabetic_predict/
├── src/                    # Training source code
│   ├── lib/               # Supporting modules
│   │   ├── dataset.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── models/
│   │   ├── optimizers.py
│   │   ├── preprocess.py
│   │   └── utils.py
│   ├── train.py           # Main training script
│   └── test.py            # Model testing script
├── weights/                # Pretrained model weights
│   ├── model_1.pth        # SE-ResNeXt50 model (fold 1)
│   └── args.txt           # Training configuration
├── demo_data/             # Sample images for testing (5 images per label)
│   ├── 0/                 # No DR
│   ├── 1/                 # Mild
│   ├── 2/                 # Moderate
│   ├── 3/                 # Severe
│   └── 4/                 # Proliferative DR
├── gradio_demo.py         # Web demo using Gradio
├── simple_inference_demo.py  # Simple command-line inference demo
├── requirements.txt        # Python dependencies
├── environment.yml         # Conda environment file
└── README.md              # This file
```

## Model

The project uses **SE-ResNeXt50_32x4d** pretrained model, which is the lightest and fastest option suitable for CPU inference.

- **Model location**: `weights/model_1.pth` (included in package)
- **Model size**: ~98MB
- **Accuracy**: ~0.91 Kappa score
- **Inference time**: ~2-5 seconds per image on CPU

## Installation

### Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate eyediabetic
pip install -r requirements.txt
```

## Usage

### 1. Web Demo (Gradio)

Launch the web interface:

```bash
python gradio_demo.py
```

Then open your browser and navigate to `http://127.0.0.1:7860`

**Features:**
- Upload retinal images
- Real-time prediction
- Visual results display

### 2. Command-Line Demo

Run inference on a single image:

```bash
python simple_inference_demo.py demo_data/0/002c21358ce6.png
```

### 3. Training Model

**Setup folder structure:**
```bash
cd src
python setup_folders.py
```

Then place your data:
- `inputs/train.csv` - Training labels
- `inputs/train_images/` - Training images (*.png)
- (Optional) `inputs/test.csv` and `inputs/test_images/` for testing

**Train the model:**
```bash
cd src
python train.py --arch se_resnext50_32x4d --train_dataset aptos2019
```

For more training options, see `src/train.py --help` or `src/README_TRAINING.md`

## Severity Levels

The model predicts 5 severity levels:

- **0**: No DR - No diabetic retinopathy
- **1**: Mild - Mild diabetic retinopathy
- **2**: Moderate - Moderate diabetic retinopathy
- **3**: Severe - Severe diabetic retinopathy
- **4**: Proliferative DR - Very severe, requires immediate treatment

## Model Path Configuration

The pretrained model is included in this package:
- **Location**: `eyediabetic_predict/weights/model_1.pth`
- **Model**: SE-ResNeXt50_32x4d (fold 1 of 5)
- **Size**: ~98MB

The code automatically searches for the model in the following locations (in order):
1. `weights/model_1.pth` (current directory - **included**)
2. `../weights/model_1.pth` (parent directory)
3. Legacy paths for backward compatibility

The model is already included in the package, so no additional setup is needed.

## Requirements

- Python 3.8+
- PyTorch 1.7+
- Gradio 3.0+ (for web demo)
- pretrainedmodels
- OpenCV
- scikit-image
- Other dependencies listed in `requirements.txt`

## Training Code Structure

The training code in `src/` is kept **unchanged** from the original Kaggle solution to:
- Maintain exact compatibility with original implementation
- Make it easy to compare and verify results
- Preserve all original functionality

**Folder structure**: The code expects Kaggle-style folder structure (see `src/README_TRAINING.md` for details). Use `src/setup_folders.py` to create the required folders.

## Notes

- This is a demo application. Results are for reference only.
- For medical diagnosis, please consult a medical specialist.
- The model is optimized for CPU inference (no GPU required).
- Inference time may vary depending on your hardware.
- Training code uses hardcoded paths - see `src/README_TRAINING.md` for setup instructions.

## License

See LICENSE file in the original repository.

## References

- APTOS 2019 Blindness Detection Competition: https://www.kaggle.com/c/aptos2019-blindness-detection
- **Author: _____**


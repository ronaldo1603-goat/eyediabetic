"""
Simple Inference Demo for APTOS 2019 Blindness Detection
Uses pretrained model directly (no fine-tuning)
Optimized for CPU inference
"""

import os
import sys
import numpy as np
import cv2
from skimage import measure
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision

try:
    import pretrainedmodels
except ImportError:
    print("Please install pretrainedmodels: pip install pretrainedmodels")
    sys.exit(1)


def scale_radius(src, img_size, padding=False):
    """Normalize image based on retinal radius"""
    x = src[src.shape[0] // 2, ...].sum(axis=1)
    r = (x > x.mean() / 10).sum() // 2
    yx = src.sum(axis=2)
    region_props = measure.regionprops((yx > yx.mean() / 10).astype('uint8'))
    yc, xc = np.round(region_props[0].centroid).astype('int')
    x1 = max(xc - r, 0)
    x2 = min(xc + r, src.shape[1] - 1)
    y1 = max(yc - r, 0)
    y2 = min(yc + r, src.shape[0] - 1)
    dst = src[y1:y2, x1:x2]
    dst = cv2.resize(dst, dsize=None, fx=img_size/(2*r), fy=img_size/(2*r))
    if padding:
        pad_x = (img_size - dst.shape[1]) // 2
        pad_y = (img_size - dst.shape[0]) // 2
        dst = np.pad(dst, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), 'constant')
    return dst


def get_model(model_name='se_resnext50_32x4d', num_outputs=1, pretrained=False):
    """Create model with architecture similar to notebook"""
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=pretrained)
    
    # Modify last layer
    if 'resnet' in model_name:
        model.avgpool = nn.AdaptiveAvgPool2d(1)
    else:
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
    
    in_features = model.last_linear.in_features
    model.last_linear = nn.Linear(in_features, num_outputs)
    
    return model


def preprocess_image(img_path, img_size=256):
    """Preprocess image similar to notebook"""
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    
    # Scale radius
    try:
        img = scale_radius(img, img_size=288, padding=False)
    except Exception as e:
        print(f"Warning: scale_radius failed, using original: {e}")
        img = cv2.resize(img, (288, 288))
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    
    # Transform (similar to validation transform in notebook)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor


def predict_diagnosis(model, img_tensor, device='cpu'):
    """Predict disease severity from image"""
    model.eval()
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        output = model(img_tensor)
        pred_value = output.item()
    
    # Convert regression output to classification (similar to notebook)
    thrs = [0.5, 1.5, 2.5, 3.5]
    if pred_value < thrs[0]:
        diagnosis = 0
    elif pred_value < thrs[1]:
        diagnosis = 1
    elif pred_value < thrs[2]:
        diagnosis = 2
    elif pred_value < thrs[3]:
        diagnosis = 3
    else:
        diagnosis = 4
    
    return diagnosis, pred_value


def load_model(model_path, device='cpu'):
    """Load pretrained model"""
    print(f"Loading model from: {model_path}")
    
    model = get_model(model_name='se_resnext50_32x4d', num_outputs=1, pretrained=False)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model


def find_model_path():
    """Find model path in various possible locations"""
    possible_paths = [
        "weights/model_1.pth",      # Current directory (included in package)
        "../weights/model_1.pth",    # Parent directory
        "../se_resnext50_32x4d_0809/model_1.pth",  # Legacy path (backward compatibility)
        "se_resnext50_32x4d_0809/model_1.pth",      # Legacy path (backward compatibility)
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "model_1.pth"),  # Absolute from project root
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def main():
    """Demo inference"""
    # Configuration
    MODEL_PATH = find_model_path()
    DEVICE = 'cpu'  # Use CPU
    
    if MODEL_PATH is None:
        print("ERROR: Model not found. Please ensure model_1.pth exists in weights/ folder")
        print("Possible locations:")
        print("  - weights/model_1.pth (current directory - included in package)")
        print("  - ../weights/model_1.pth (parent directory)")
        return
    
    # Load model
    model = load_model(MODEL_PATH, device=DEVICE)
    
    # Test with an image - use demo_data if available, or command line argument
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
    else:
        # Try to use demo_data
        demo_data_paths = [
            "demo_data/0/002c21358ce6.png",
            "demo_data/1/0024cdab0c1e.png",
            "demo_data/2/000c1434d8d7.png",
        ]
        test_image_path = None
        for path in demo_data_paths:
            if os.path.exists(path):
                test_image_path = path
                break
    
    if test_image_path is None or not os.path.exists(test_image_path):
        print(f"\nDemo: Please provide an image path to test")
        print(f"Usage: python simple_inference_demo.py <image_path>")
        print(f"Or place images in demo_data/ folder")
        return
    
    # Preprocess and predict
    print(f"\nProcessing image: {test_image_path}")
    img_tensor = preprocess_image(test_image_path)
    
    diagnosis, raw_value = predict_diagnosis(model, img_tensor, device=DEVICE)
    
    # Results
    diagnosis_labels = {
        0: "No DR (No disease)",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative DR (Very severe)"
    }
    
    print(f"\n{'='*50}")
    print(f"PREDICTION RESULT:")
    print(f"  - Raw value: {raw_value:.4f}")
    print(f"  - Diagnosis: {diagnosis} - {diagnosis_labels[diagnosis]}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

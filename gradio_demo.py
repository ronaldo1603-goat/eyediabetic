"""
Gradio Web Demo for APTOS 2019 Blindness Detection
Uses pretrained model to predict diabetic retinopathy severity levels
"""

import os
import numpy as np
import cv2
from skimage import measure
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import gradio as gr

try:
    import pretrainedmodels
except ImportError:
    print("Please install pretrainedmodels: pip install pretrainedmodels")
    exit(1)


# ========== Model and Preprocessing Functions ==========

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


def preprocess_image_from_file(img_path, img_size=256):
    """Preprocess image from file path"""
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
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor


def preprocess_image_from_pil(pil_image):
    """Preprocess image from PIL Image (from Gradio)"""
    # Convert PIL to numpy array
    img = np.array(pil_image)
    
    # Convert RGB to BGR for OpenCV processing
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img
    
    # Scale radius
    try:
        img_processed = scale_radius(img_bgr, img_size=288, padding=False)
    except Exception as e:
        print(f"Warning: scale_radius failed, using resized: {e}")
        img_processed = cv2.resize(img_bgr, (288, 288))
    
    # Convert back to RGB
    img_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    img_tensor = transform(img_pil).unsqueeze(0)
    return img_tensor


def predict_diagnosis(model, img_tensor, device='cpu'):
    """Predict disease severity from image"""
    model.eval()
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        output = model(img_tensor)
        pred_value = output.item()
    
    # Convert regression output to classification
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


# ========== Global Model Variable ==========
MODEL = None
DEVICE = 'cpu'


def initialize_model():
    """Initialize model when app starts"""
    global MODEL, DEVICE
    
    # Try to find model in various possible locations
    MODEL_PATH = None
    possible_paths = [
        "weights/model_1.pth",      # Current directory (included in package)
        "../weights/model_1.pth",   # Parent directory
        "../se_resnext50_32x4d_0809/model_1.pth",  # Legacy path (backward compatibility)
        "se_resnext50_32x4d_0809/model_1.pth",      # Legacy path (backward compatibility)
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "model_1.pth"),  # Absolute from project root
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            MODEL_PATH = path
            break
    
    if MODEL_PATH is None:
        raise FileNotFoundError(f"Model not found. Please ensure model_1.pth exists in weights/ folder")
    
    MODEL = load_model(MODEL_PATH, device=DEVICE)
    return "Model is ready!"


# ========== Gradio Prediction Function ==========

def predict(image):
    """Function for Gradio to call when Predict button is clicked"""
    global MODEL
    
    if MODEL is None:
        return "❌ Error: Model not loaded. Please refresh the page.", None
    
    if image is None:
        return "❌ Please select an image before predicting.", None
    
    try:
        # Preprocess image
        img_tensor = preprocess_image_from_pil(image)
        
        # Predict
        diagnosis, raw_value = predict_diagnosis(MODEL, img_tensor, device=DEVICE)
        
        # Define labels
        diagnosis_labels = {
            0: "No DR (No Diabetic Retinopathy)",
            1: "Mild",
            2: "Moderate",
            3: "Severe",
            4: "Proliferative DR (Very Severe - Immediate Treatment Required)"
        }
        
        diagnosis_descriptions = {
            0: "Healthy eye, no signs of diabetic retinopathy.",
            1: "Mild signs of diabetic retinopathy. Regular monitoring recommended.",
            2: "Moderate severity. Please consult a specialist.",
            3: "Severe condition. Immediate treatment required.",
            4: "Very severe condition. Urgent medical intervention needed."
        }
        
        # Create result display
        result_text = f"""
# 🔍 PREDICTION RESULT

## Severity Level: **{diagnosis} - {diagnosis_labels[diagnosis]}**

**Predicted Value:** {raw_value:.4f}

**Description:** {diagnosis_descriptions[diagnosis]}

---
*Note: This is an automated prediction result. Please consult a medical specialist for accurate diagnosis.*
        """
        
        return result_text, image
        
    except Exception as e:
        return f"❌ Error during processing: {str(e)}", None


# ========== Gradio Interface ==========

def create_interface():
    """Create Gradio interface"""
    
    # Initialize model
    try:
        initialize_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("App will still run but will show error when predicting.")
    
    with gr.Blocks(title="APTOS 2019 Blindness Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 👁️ Diabetic Retinopathy Detection
            
            This application uses Deep Learning to predict the severity level of diabetic retinopathy from retinal images.
            
            **Usage Instructions:**
            1. Select a retinal image using the "Choose Image" button below
            2. Click the "Predict" button to start prediction
            3. Results will be displayed in the center of the screen
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Image")
                image_input = gr.Image(
                    type="pil",
                    label="Select retinal image",
                    height=400
                )
                
                predict_btn = gr.Button(
                    "🔮 Predict",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Prediction Results")
                result_output = gr.Markdown(
                    value="Results will be displayed here after clicking the Predict button...",
                    elem_classes=["result-box"]
                )
                
                image_output = gr.Image(
                    label="Processed Image",
                    visible=True,
                    height=400
                )
        
        # Event handler
        predict_btn.click(
            fn=predict,
            inputs=image_input,
            outputs=[result_output, image_output]
        )
        
        gr.Markdown(
            """
            ---
            ### ℹ️ Information about severity levels:
            - **0**: No DR - No disease
            - **1**: Mild
            - **2**: Moderate
            - **3**: Severe
            - **4**: Proliferative DR - Very severe
            
            **Note:** This is a demo application, results are for reference only.
            """
        )
    
    return demo


# ========== Main ==========

if __name__ == "__main__":
    # Create and launch interface
    demo = create_interface()
    
    # Launch with options
    demo.launch(
        server_name="0.0.0.0",  # Allow access from local network
        server_port=7860,       # Default Gradio port
        share=False,            # Set True to create public link
        show_error=True
    )


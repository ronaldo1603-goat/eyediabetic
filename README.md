# eyediabetic
eyediabetic
#  Eyediabetic AI Project
**AI-powered Retinal Lesion Detection for Diabetic Patients**

---

##  Overview
**Eyediabetic AI** is a deep learning–based system designed to automatically detect and classify diabetic retinopathy lesions from retinal fundus images.  
The goal is to assist ophthalmologists in early diagnosis and reduce vision loss risks among diabetic patients.

---

##  Features
- Automated retinal image preprocessing (contrast enhancement, normalization)
- CNN-based lesion detection and severity classification
- Visualization of heatmaps (Grad-CAM) for model interpretability
- Support for batch image analysis and report generation

---

##  Model Architecture
- **Base Model:** ResNet50 / EfficientNet (transfer learning)  
- **Framework:** PyTorch / TensorFlow  
- **Input:** Fundus images (.jpg/.png)  
- **Output:** Classification: *No DR, Mild, Moderate, Severe, Proliferative*

---
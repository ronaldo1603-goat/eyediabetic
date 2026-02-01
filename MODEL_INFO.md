# Model Information

## Pretrained Model

**Model**: SE-ResNeXt50_32x4d  
**Location**: `weights/model_1.pth`  
**Size**: ~98MB  
**Fold**: 1 of 5 (for demo purposes, only fold 1 is included)

## Model Details

- **Architecture**: SE-ResNeXt50_32x4d
- **Training**: 30 epochs on APTOS 2019 dataset
- **Accuracy**: ~0.91 Quadratic Weighted Kappa
- **Input size**: 256x256
- **Output**: Regression value (converted to 0-4 classification)

## Model Files Included

- `model_1.pth` - Pretrained weights (fold 1)
- `args.txt` - Training arguments and configuration

## Using Other Folds

If you have access to other folds (model_2.pth to model_5.pth), you can:
1. Copy them to `weights/` folder
2. Modify the code to average predictions from all 5 folds for better accuracy

## Model Loading

The demo code automatically finds the model in:
- Current directory: `weights/model_1.pth` ✅ (included)
- Parent directory: `../weights/model_1.pth`
- Legacy paths for backward compatibility

No additional configuration needed - the model is ready to use!


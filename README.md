# DEReD: Complete Implementation Guide

![DEReD Banner](https://img.shields.io/badge/DEReD-Complete%20Implementation-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange) ![CUDA](https://img.shields.io/badge/CUDA-11.8+-red)

**Fully Self-Supervised Depth Estimation from Defocus Clue - Complete Reproduction Guide**

This repository provides a complete, step-by-step implementation of the DEReD (Depth Estimation via Reconstructing Defocus Image) paper from CVPR 2023. This guide will take you from a fresh system to fully working depth estimation with comprehensive evaluation results.

## 📋 Table of Contents

- [Overview](#-overview)
- [Prerequisites](#-prerequisites)
- [Installation Guide](#-installation-guide)
- [Dataset Preparation](#-dataset-preparation)
- [Training & Evaluation](#-training--evaluation)
- [Results & Outputs](#-results--outputs)
- [Troubleshooting](#-troubleshooting)
- [Advanced Usage](#-advanced-usage)
- [Citation](#-citation)

## 🎯 Overview

### What is DEReD?

DEReD is a self-supervised neural network that estimates depth maps and generates All-in-Focus (AIF) images from focal stack inputs. Unlike traditional methods, it requires **no ground-truth depth data** during training, making it highly practical for real-world applications.

### Key Features of This Implementation

- ✅ **Complete DAIF-Net architecture** with dual-branch design
- ✅ **Both depth maps AND AIF image generation**
- ✅ **Proper training/testing split** with comprehensive evaluation
- ✅ **Practice counter and progress tracking**
- ✅ **Automated focal stack generation** from RGB-D data
- ✅ **Standard depth estimation metrics** (RMSE, AbsRel, δ1, δ2, δ3)
- ✅ **Visual comparison tools** and error analysis
- ✅ **Windows/Linux compatibility** with CUDA support

### Expected Results

After following this guide, you will have:
- Trained depth estimation model
- Generated depth maps for test images
- All-in-Focus image reconstructions
- Comprehensive evaluation metrics
- Practice tracking and progress reports

## 🔧 Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with 6GB+ VRAM (RTX 3060 or better recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ free space
- **OS**: Windows 10/11 or Linux

### Software Requirements

- **Python**: 3.8 or 3.9 (3.9 recommended)
- **CUDA**: 11.8 or 12.x
- **Git**: For repository cloning
- **Anaconda/Miniconda**: For environment management

### Knowledge Prerequisites

- Basic Python programming
- Familiarity with command line/terminal
- Basic understanding of deep learning concepts (helpful but not required)

## 🚀 Installation Guide

### Step 1: Environment Setup

#### Option A: Windows with Anaconda Prompt (Recommended)

```bash
# Open Anaconda Prompt (not Git Bash)
conda create -n dered python=3.9
conda activate dered

# Verify activation
python --version  # Should show Python 3.9.x
```

#### Option B: Linux/WSL

```bash
conda create -n dered python=3.9
conda activate dered
python --version
```

### Step 2: Clone Repository

```bash
git clone https://github.com/Ehzoahis/DEReD.git
cd DEReD
```

### Step 3: Install Dependencies

#### Install PyTorch with CUDA

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1 (if you have newer drivers)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA installation
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

#### Install Other Dependencies

```bash
# Core dependencies
pip install numpy scipy matplotlib tqdm pillow h5py

# Optional (for logging and visualization)
pip install wandb tensorboard

# For dataset processing
pip install opencv-python
```

### Step 4: Setup Gaussian PSF Module (Optional)

If the `gauss_psf` directory exists:

```bash
cd gauss_psf
python setup.py install
cd ..
```

### Step 5: Verify Installation

Create and run `test_setup.py`:

```python
import torch
import numpy as np
import cv2
from PIL import Image

print("🔍 Testing DEReD Setup...")
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("✅ All dependencies working!")
```

Run with: `python test_setup.py`

## 📊 Dataset Preparation

### Option 1: NYUv2 Dataset (Recommended)

#### Download NYUv2 Dataset

1. **Download the dataset**:
   - Get `nyu_depth_v2_labeled.mat` from [NYU Depth V2 website](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
   - Place it in your project root directory

2. **Extract RGB and Depth data**:

Create `load_mat.py`:

```python
import h5py
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def extract_nyuv2_data():
    print("🔄 Extracting NYUv2 data...")
    
    with h5py.File('nyu_depth_v2_labeled.mat', 'r') as f:
        images = f['images']  # (1449, 3, 640, 480)
        depths = f['depths']  # (1449, 640, 480)
        
        total = images.shape[0]
        print(f"Total images: {total}")
        
        # Create directories
        os.makedirs('data/NYUv2/train_rgb', exist_ok=True)
        os.makedirs('data/NYUv2/train_depth', exist_ok=True)
        
        # Split: first 1000 for training, rest for testing
        splits = [
            ('train', 0, 1000),
            ('test', 1000, total)
        ]
        
        for split_name, start_idx, end_idx in splits:
            os.makedirs(f'data/NYUv2/{split_name}_rgb', exist_ok=True)
            os.makedirs(f'data/NYUv2/{split_name}_depth', exist_ok=True)
            
            for i in tqdm(range(start_idx, end_idx), desc=f'Extracting {split_name}'):
                # Extract and convert RGB: (3, 640, 480) -> (640, 480, 3)
                rgb = np.array(images[i]).transpose(1, 2, 0)
                depth = np.array(depths[i])
                
                # Convert to proper formats
                rgb_img = Image.fromarray(rgb.astype(np.uint8))
                depth_img = Image.fromarray((depth * 1000).astype(np.uint16))
                
                # Save
                rgb_img.save(f'data/NYUv2/{split_name}_rgb/{i:04d}.png')
                depth_img.save(f'data/NYUv2/{split_name}_depth/{i:04d}.png')
        
        print("✅ NYUv2 data extraction complete!")

if __name__ == "__main__":
    extract_nyuv2_data()
```

Run: `python load_mat.py`

#### Generate Focal Stacks

Create `generate_focal_stacks.py`:

```python
import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

def calculate_defocus_blur(depth, focus_distance, focal_length=50e-3, f_number=8.0):
    """Calculate defocus blur using thin lens equation"""
    depth_m = depth / 1000.0  # Convert to meters
    focus_distance_m = focus_distance
    
    # Circle of confusion calculation
    coc = np.abs(focal_length * (focus_distance_m - depth_m) / 
                (f_number * depth_m * (focus_distance_m - focal_length)))
    
    # Convert to pixel blur radius
    sensor_size = 23.5e-3  # 35mm sensor width
    pixel_size = sensor_size / 640
    blur_radius = coc / pixel_size
    
    return np.clip(blur_radius, 0, 20)

def apply_gaussian_blur(image, blur_map):
    """Apply spatially varying Gaussian blur"""
    result = np.zeros_like(image)
    max_blur = np.max(blur_map)
    
    if max_blur < 0.5:
        return image
    
    blur_levels = np.linspace(0, max_blur, 15)
    
    for i in range(1, len(blur_levels)):
        mask = (blur_map >= blur_levels[i-1]) & (blur_map < blur_levels[i])
        if np.sum(mask) == 0:
            continue
        
        sigma = blur_levels[i]
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            kernel_size = 3
        
        if sigma > 0.5:
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        else:
            blurred = image
        
        mask_3d = np.stack([mask] * 3, axis=2)
        result += blurred * mask_3d
    
    # Handle unblurred regions
    unblurred_mask = blur_map < blur_levels[1]
    unblurred_mask_3d = np.stack([unblurred_mask] * 3, axis=2)
    result += image * unblurred_mask_3d
    
    return result.astype(np.uint8)

def generate_focal_stacks():
    """Generate focal stacks from RGB-D data"""
    focus_distances = [1, 3, 5, 7, 9]  # meters
    
    for split in ['train', 'test']:
        rgb_dir = f'data/NYUv2/{split}_rgb'
        depth_dir = f'data/NYUv2/{split}_depth'
        output_dir = f'data/NYUv2/{split}_fs5'
        
        os.makedirs(output_dir, exist_ok=True)
        
        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
        
        print(f"🔄 Generating {split} focal stacks...")
        
        for i, rgb_file in enumerate(tqdm(rgb_files[:100])):  # Limit for demo
            try:
                # Load data
                rgb_path = os.path.join(rgb_dir, rgb_file)
                depth_path = os.path.join(depth_dir, rgb_file)
                
                if not os.path.exists(depth_path):
                    continue
                
                rgb = cv2.imread(rgb_path)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (320, 240))
                
                depth = np.array(Image.open(depth_path), dtype=np.float32)
                depth = cv2.resize(depth, (320, 240))
                
                # Generate focal stack
                sample_dir = os.path.join(output_dir, f'sample_{i:04d}')
                os.makedirs(sample_dir, exist_ok=True)
                
                for focus_dist in focus_distances:
                    blur_map = calculate_defocus_blur(depth, focus_dist)
                    defocused = apply_gaussian_blur(rgb, blur_map)
                    
                    focus_img = Image.fromarray(defocused)
                    focus_img.save(os.path.join(sample_dir, f'focus_{focus_dist:.0f}m.png'))
                
                # Save depth
                depth_resized = cv2.resize(depth, (320, 240))
                depth_img = Image.fromarray(depth_resized.astype(np.uint16))
                depth_img.save(os.path.join(sample_dir, 'depth.png'))
                
            except Exception as e:
                print(f"Error processing {rgb_file}: {e}")
                continue
        
        print(f"✅ {split} focal stacks generated!")

if __name__ == "__main__":
    generate_focal_stacks()
```

Run: `python generate_focal_stacks.py`

### Option 2: Use Pre-generated Data

If you have access to pre-generated focal stacks, place them in:
```
data/NYUv2/
├── train_fs5/
│   ├── sample_0000/
│   │   ├── focus_1m.png
│   │   ├── focus_3m.png
│   │   ├── focus_5m.png
│   │   ├── focus_7m.png
│   │   ├── focus_9m.png
│   │   └── depth.png
│   └── ...
└── test_fs5/
    └── ...
```

## 🏋️ Training & Evaluation

### Quick Start Training

For a quick test run:

```bash
python scripts/train_complete_with_evaluation.py --use_cuda --BS 4 --epochs 5 --name quick_test
```

### Full Training with Evaluation

```bash
python scripts/train_complete_with_evaluation.py \
    --use_cuda \
    --BS 4 \
    --epochs 15 \
    --dataset NYUv2 \
    --data_path ./data/NYUv2 \
    --name full_training
```

### Training Parameters Explained

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `--use_cuda` | Enable GPU acceleration | Always use if GPU available |
| `--BS` | Batch size | 4-8 (depends on GPU memory) |
| `--epochs` | Number of training epochs | 15-25 for full training |
| `--dataset` | Dataset type | NYUv2 or DefocusNet |
| `--data_path` | Path to dataset | ./data/NYUv2 |
| `--name` | Experiment name | Choose descriptive name |

### What Happens During Training

1. **Practice Tracking**: Automatically counts and tracks your training runs
2. **Train/Test Split**: Splits data into 80% training, 20% testing
3. **Dual Training**: Trains both depth and AIF prediction simultaneously
4. **Evaluation**: Tests on both training and testing sets each epoch
5. **Best Model Saving**: Automatically saves the best performing model
6. **Visual Generation**: Creates depth maps, AIF images, and comparisons

## 📊 Results & Outputs

### Generated Files Structure

After training, you'll find:

```
DEReD/
├── experiments/
│   └── best_model_practice_X.pth      # Trained model
├── results/
│   └── practice_X/
│       ├── train_depth/               # Training depth predictions
│       ├── train_aif/                 # Training AIF predictions
│       ├── test_depth/                # Testing depth predictions
│       ├── test_aif/                  # Testing AIF predictions
│       └── comparisons/               # Side-by-side visualizations
├── practice_log.json                 # Complete practice history
└── practice_summary_report.txt       # Detailed analysis
```

### Evaluation Metrics

For both training and testing sets, you'll get:

#### Depth Estimation Metrics
- **RMSE**: Root Mean Square Error (lower is better)
- **AbsRel**: Absolute Relative Error (lower is better)
- **δ1**: Percentage of pixels with error < 1.25 (higher is better)
- **δ2**: Percentage of pixels with error < 1.25² (higher is better)
- **δ3**: Percentage of pixels with error < 1.25³ (higher is better)

#### Expected Performance
Based on the original paper, you should expect:

**NYUv2 Dataset:**
- RMSE: 0.8-1.2 meters
- AbsRel: 0.15-0.25
- δ1: 0.7-0.8
- δ2: 0.9-0.95
- δ3: 0.97-0.99

### Visual Results

The system generates:
1. **Depth Maps**: Grayscale images showing predicted depth
2. **AIF Images**: Sharp, all-in-focus reconstructions
3. **Error Maps**: Visual representation of prediction errors
4. **Comparison Grids**: Side-by-side input/prediction/ground-truth

### Practice Tracking

Each training run is automatically tracked:
- **Practice Counter**: Shows how many times you've trained
- **Progress Analysis**: Compares current vs previous results
- **Best Performance**: Tracks your best results across all practices
- **Detailed History**: Stores complete training logs

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Reduce batch size
python scripts/train_complete_with_evaluation.py --BS 2

# Or use CPU (slower)
python scripts/train_complete_with_evaluation.py  # Remove --use_cuda
```

#### 2. Import Errors

**Error**: `ModuleNotFoundError: No module named 'model'`

**Solutions**:
```bash
# Ensure you're in the correct directory
cd DEReD

# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# On Windows
set PYTHONPATH=%PYTHONPATH%;%CD%
```

#### 3. Dataset Not Found

**Error**: `Dataset: 0 samples loaded`

**Solutions**:
1. Verify data extraction completed:
   ```bash
   ls data/NYUv2/train_fs5/
   ```
2. Re-run focal stack generation:
   ```bash
   python generate_focal_stacks.py
   ```

#### 4. Slow Training

**Solutions**:
- Ensure CUDA is properly installed and detected
- Reduce image resolution in dataset loader
- Use fewer focal stack images per sample

#### 5. Windows-Specific Issues

**Conda Activation Error**:
- Use Anaconda Prompt instead of Git Bash
- Or configure Git Bash for conda:
  ```bash
  echo '. ${HOME}/anaconda3/etc/profile.d/conda.sh' >> ~/.bashrc
  source ~/.bashrc
  ```

### Debugging Tips

1. **Check GPU usage**:
   ```bash
   nvidia-smi
   ```

2. **Monitor training**:
   ```bash
   watch -n 1 nvidia-smi  # Linux
   # Or check Task Manager GPU usage on Windows
   ```

3. **Test with dummy data**:
   If your dataset isn't working, the code will automatically fall back to dummy data for testing.

## 🎛️ Advanced Usage

### Custom Datasets

To use your own dataset:

1. **Organize your data**:
   ```
   your_data/
   ├── train_fs5/
   │   ├── sample_0000/
   │   │   ├── focus_1m.png
   │   │   ├── focus_3m.png
   │   │   └── depth.png
   │   └── ...
   └── test_fs5/
       └── ...
   ```

2. **Modify the dataset class** in `train_complete_with_evaluation.py`:
   - Update focus distances
   - Adjust normalization values
   - Change image dimensions

### Hyperparameter Tuning

Key parameters to experiment with:

```python
# In the training script
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Learning rate
total_loss = depth_loss + 0.5 * aif_loss  # Loss weights
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
```

### Model Architecture Modifications

To modify the DAIF-Net:

1. **Change network depth**:
   ```python
   # Add more convolutional layers in CompleteDAIFNet
   ```

2. **Adjust channel dimensions**:
   ```python
   # Modify the channel numbers in conv layers
   ```

3. **Add attention mechanisms**:
   ```python
   # Implement attention blocks between encoder and decoder
   ```

### Evaluation on Different Metrics

Add custom evaluation metrics in the `calculate_depth_metrics` function:

```python
def calculate_depth_metrics(pred_depth, gt_depth, max_depth=10.0):
    # Add your custom metrics here
    custom_metric = your_calculation(pred_depth, gt_depth)
    
    return {
        'rmse': rmse.item(),
        'abs_rel': abs_rel.item(),
        'custom_metric': custom_metric,
        # ... other metrics
    }
```

## 🎓 Understanding the Results

### Interpreting Depth Maps

- **Bright areas**: Closer objects (higher depth values)
- **Dark areas**: Farther objects (lower depth values)
- **Sharp boundaries**: Good depth discontinuity detection
- **Smooth gradients**: Good depth estimation in continuous surfaces

### Interpreting AIF Images

- **Sharp details**: Good all-in-focus reconstruction
- **Minimal blur**: Effective defocus removal
- **Preserved textures**: Good image quality maintenance
- **Consistent colors**: Proper image reconstruction

### Training Progress Indicators

**Good Training**:
- Decreasing loss over epochs
- Similar train/test performance
- Improving depth metrics (δ1, δ2, δ3 increasing)

**Problematic Training**:
- Loss plateauing early
- Large train/test gap (overfitting)
- Unstable loss curves

## 📚 Additional Resources

### Related Papers

1. **Original DEReD Paper**: [Fully Self-Supervised Depth Estimation from Defocus Clue](https://arxiv.org/abs/2303.10752)
2. **DefocusNet**: [Defocus Deblurring Using Dual-Pixel Data](https://github.com/dvl-tum/defocus-net)
3. **NYUv2 Dataset**: [Indoor Segmentation and Support Inference from RGBD Images](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

### Useful Tools

- **Weights & Biases**: For advanced training monitoring
- **TensorBoard**: For visualization
- **OpenCV**: For image processing
- **Matplotlib**: For plotting results

### Community

- **GitHub Issues**: Report bugs and ask questions
- **Research Papers**: Check recent depth estimation papers for improvements
- **PyTorch Forums**: For general PyTorch questions

## 🔗 Citation

If you use this implementation in your research, please cite:

```bibtex
@article{si2023fully,
  title={Fully Self-Supervised Depth Estimation from Defocus Clue},
  author={Si, Haozhe and Zhao, Bin and Wang, Dong and Gao, Yupeng and Chen, Mulin and Wang, Zhigang and Li, Xuelong},
  journal={arXiv preprint arXiv:2303.10752},
  year={2023}
}
```

## 📄 License

This project follows the same license as the original DEReD repository. Please check the original repository for license details.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

If you encounter issues:

1. **Check the troubleshooting section** above
2. **Search existing GitHub issues**
3. **Create a new issue** with:
   - Your system specifications
   - Complete error messages
   - Steps to reproduce the problem

---

**🎉 Congratulations!** You now have a complete guide to reproduce the DEReD paper with full evaluation capabilities. The implementation provides both depth maps and AIF images with comprehensive training/testing analysis and practice tracking.

**Happy depth estimation!** 🚀

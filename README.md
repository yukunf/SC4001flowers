# SC4001 Project- Flower Classification

---

## Environment Setup

Please run following code to run the code

```
conda create sc4001 python=3.12
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip3 install scipy
pip3 install transformers
```

Depends on your device you may choose a different version of cuda of torch.

---

## Project Overview

This project implements and compares various ResNet architectures for flower classification on the challenging Oxford Flowers 102 dataset. The dataset contains 102 flower species native to the United Kingdom with significant variations in scale, pose, lighting conditions, and intra-class diversity. The project explores different transfer learning strategies, architectural depths, and data efficiency through comprehensive experiments.

### Dataset Information

- **Dataset**: [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- **PyTorch Integration**: [TorchVision Flowers102](https://pytorch.org/vision/main/generated/torchvision.datasets.Flowers102.html)
- **Total Images**: 8,189 images across 102 flower species
- **Split**: Training (1,020), Validation (1,020), Test (6,149)

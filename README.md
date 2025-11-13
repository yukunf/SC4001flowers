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

_Note: Depending on your device, you may need to choose a different CUDA version for PyTorch._

---

## Project Overview

This project implements and compares various ResNet architectures for flower classification on the challenging Oxford Flowers 102 dataset. The dataset contains 102 flower species native to the United Kingdom with significant variations in scale, pose, lighting conditions, and intra-class diversity. The project explores different transfer learning strategies, architectural depths, and data efficiency through comprehensive experiments.

### Dataset Information

- **Dataset**: [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- **PyTorch Integration**: [TorchVision Flowers102](https://pytorch.org/vision/main/generated/torchvision.datasets.Flowers102.html)
- **Total Images**: 8,189 images across 102 flower species
- **Split**: Training (1,020), Validation (1,020), Test (6,149)

## Final Results Summary

### Model Performance Comparison

| Model                          | Test Accuracy | Key Characteristics             |
| ------------------------------ | ------------- | ------------------------------- |
| **ResNet-34 Frozen**           | 90.00%        | Traditional CNN baseline        |
| **CLIP + LoRA**                | 92.00%        | Parameter-efficient fine-tuning |
| **CLIP + LoRA + MixUp**        | **91.09%**    | **Best overall performance**    |
| **CLIP + LoRA + Triplet Loss** | 90.93%        | Metric learning approach        |
| **CLIP + VPT**                 | 87.98%        | Visual prompt tuning            |

### Key Findings

1. **CLIP Superiority**: CLIP-based models consistently outperformed ResNet baselines
2. **Parameter Efficiency**: LoRA achieved 92% accuracy with only ~3% trainable parameters
3. **Data Efficiency**: CLIP + LoRA showed exceptional few-shot learning capabilities
4. **Regularization Benefits**: MixUp provided the most effective performance improvement
5. **Architecture Choice**: Vision-language models proved more effective than traditional CNNs for fine-grained classification

### Few-Shot Learning Performance

| Data Fraction | ResNet-34 | CLIP + LoRA |
| ------------- | --------- | ----------- |
| 10%           | 18.70%    | **38.61%**  |
| 20%           | 38.41%    | **64.55%**  |
| 50%           | 64.48%    | **84.06%**  |
| 100%          | 87.10%    | **91.12%**  |

## Project Structure

Each folder contains detailed implementations and experiments:

### [3.1_resnet](3.1_resnet/) - Traditional CNN Baselines

- ResNet-18 and ResNet-34 implementations
- Transfer learning with frozen stages
- Baseline performance establishment

### [3.2_clip](3.2_clip/) - CLIP with LoRA Fine-tuning

- Vision-language model implementation
- Parameter-efficient LoRA adaptation
- Core CLIP experimentation

### [3.3_few_shots_comparison](3.3_few_shots_comparison/) - Data Efficiency Analysis

- Few-shot learning experiments
- ResNet-34 vs CLIP data efficiency comparison
- Comprehensive performance scaling analysis

### [3.4_clip_addi_tasks](3.4_clip_addi_tasks/) - Advanced Techniques

- MixUp data augmentation
- Triplet Loss for embedding learning
- Visual Prompt Tuning (VPT)

## Execution Guide

Each folder contains its own README with detailed instructions. Generally:

1. **Start with data preprocessing** (handled automatically in notebooks)
2. **Run baseline experiments** (ResNet first, then CLIP)
3. **Proceed to advanced techniques** (MixUp, Triplet Loss, VPT)
4. **Execute comparison analyses** (few-shot learning, performance comparisons)

## Technical Notes

- All experiments use random seed 42 for reproducibility
- Early stopping and learning rate scheduling employed throughout
- Mixed precision training used where supported
- Consistent data augmentation applied across all experiments
- Model checkpoints saved for best performing configurations

## References

Based on techniques from:

- CLIP: Radford et al. "Learning Transferable Visual Models from Natural Language Supervision"
- LoRA: Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models"
- MixUp: Zhang et al. "mixup: Beyond Empirical Risk Minimization"

_Refer to individual folder README files for detailed implementation instructions and specific results._

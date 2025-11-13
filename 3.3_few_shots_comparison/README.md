# Few-Shot Learning Comparison

This folder contains the implementation and analysis of few-shot learning experiments comparing ResNet-34 and CLIP models on the Oxford Flowers-102 dataset.

## File Structure

```
3.3_few_shots_comparison/
├── ResNet34_frozen_few_shot_analysis/
│   ├── resnet34_few_shot.ipynb          # ResNet-34 few-shot experiments
│   ├── resnet34_few_shot_analysis.png   # Performance visualization
│   └── resnet34_few_shot_results.csv    # Experimental results
└── clip_few_shot.ipynb                  # CLIP few-shot experiments
```

## Notebook Descriptions

### 1. resnet34_few_shot.ipynb

**Purpose**: Few-shot learning analysis with ResNet-34 architecture

**Contents**:

- Complete ResNet-34 implementation with frozen stages
- Systematic few-shot experiments across data fractions: 10%, 20%, 30%, 50%, 75%, 100%
- Training with early stopping and learning rate scheduling
- Performance visualization and efficiency analysis

**Key Features**:

- Transfer learning with ImageNet-pretrained ResNet-34
- First three stages frozen for parameter efficiency
- Early stopping to prevent overfitting on small datasets
- Comprehensive performance metrics and visualizations

### 2. clip_few_shot.ipynb

**Purpose**: Few-shot learning analysis with CLIP + LoRA architecture

**Contents**:

- CLIP model with LoRA (Low-Rank Adaptation) fine-tuning
- Few-shot experiments matching ResNet-34 data fractions
- Parameter-efficient training with frozen backbone
- Comparative performance analysis

**Key Features**:

- CLIP vision-language model with LoRA adapters
- Text-image alignment through contrastive learning
- Minimal trainable parameters (LoRA + classification head)
- Efficient adaptation to small datasets

## Experimental Design

Both notebooks follow the same experimental protocol:

**Data Fractions**: 10%, 20%, 30%, 50%, 75%, 100% of training data
**Training Strategy**: Early stopping with patience, learning rate scheduling
**Evaluation**: Consistent testing on full test set (6,149 images)
**Reproducibility**: Fixed random seed (42) for all experiments

## Results from Notebook Implementations

### ResNet-34 Few-Shot Results

| Data % | Samples | Val Accuracy | Test Accuracy |
| ------ | ------- | ------------ | ------------- |
| 10.0%  | 102     | 21.37%       | 18.70%        |
| 20.0%  | 204     | 37.35%       | 38.41%        |
| 30.0%  | 306     | 51.76%       | 50.45%        |
| 50.0%  | 510     | 69.12%       | 64.48%        |
| 75.0%  | 765     | 85.00%       | 81.49%        |
| 100.0% | 1020    | 89.80%       | 87.10%        |

**Key Insights**:

- **Data Efficiency**: Achieves 74.0% of full performance with only 50% data
- **Rapid Scaling**: Performance more than doubles from 10% to 20% data
- **Strong Baseline**: 18.70% accuracy with only 102 training images

### CLIP + LoRA Few-Shot Results

| Data % | Samples | Val Accuracy | Test Accuracy |
| ------ | ------- | ------------ | ------------- |
| 10.0%  | 102     | 42.25%       | 38.61%        |
| 20.0%  | 204     | 70.00%       | 64.55%        |
| 30.0%  | 306     | 73.73%       | 71.65%        |
| 50.0%  | 510     | 84.80%       | 84.06%        |
| 75.0%  | 765     | 89.22%       | 90.01%        |
| 100.0% | 1020    | 91.18%       | 91.12%        |

**Key Insights**:

- **Superior Few-Shot**: Outperforms ResNet-34 across all data fractions
- **High Efficiency**: 90.01% accuracy with only 75% training data
- **Stable Convergence**: Consistent performance gains with increasing data

## Comparative Analysis

**Performance Advantage**: CLIP + LoRA consistently outperforms ResNet-34 in low-data regimes:

- **10% data**: CLIP achieves 38.61% vs ResNet's 18.70% (+106% relative improvement)
- **20% data**: CLIP achieves 64.55% vs ResNet's 38.41% (+68% relative improvement)

**Data Efficiency**: CLIP reaches ResNet's full-data performance (87.10%) with only 50-75% of training data

## Notes

- All experiments use consistent preprocessing and data augmentation
- Random seed 42 ensures reproducibility across both implementations
- Early stopping employed to prevent overfitting on small datasets
- Both models use parameter-efficient strategies (frozen stages for ResNet, LoRA for CLIP)
- Training conducted on CPU/GPU as available with mixed precision where supported

## Requirements

Key dependencies:

- PyTorch
- TorchVision
- Transformers (for CLIP)
- Matplotlib, Seaborn for visualization
- scikit-learn for metrics
- tqdm for progress bars

## Execution

Run the notebooks in any order:

```bash
jupyter notebook resnet34_few_shot.ipynb
jupyter notebook clip_few_shot.ipynb
```

Each notebook contains complete self-contained implementations with data loading, model training, and evaluation.

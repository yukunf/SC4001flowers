# CLIP Additional Tasks

This folder contains implementations of advanced training techniques applied to CLIP models for flower classification, including MixUp data augmentation, Triplet Loss, and Visual Prompt Tuning (VPT).

## File Structure

```
3.4_clip_addi_tasks/
├── clip_mixup.ipynb          # CLIP with MixUp data augmentation
├── clip_triplet_loss.ipynb   # CLIP with Triplet Loss
└── clip_vpt.ipynb           # CLIP with Visual Prompt Tuning
```

## Notebook Descriptions

### 1. clip_mixup.ipynb

**Purpose**: Implementation of MixUp data augmentation with CLIP + LoRA

**Contents**:

- CLIP model with LoRA fine-tuning
- MixUp data augmentation implementation
- Linear interpolation of images and labels
- Beta distribution for mixing coefficient sampling
- Combined loss computation for mixed samples

**Key Features**:

- **MixUp Implementation**: Creates virtual training samples by blending pairs of images and their labels
- **Beta Distribution**: Uses α=0.2 for mild mixing strength
- **Loss Adaptation**: Modified cross-entropy loss for soft labels
- **Regularization**: Improves generalization through data-space interpolation

### 2. clip_triplet_loss.ipynb

**Purpose**: Integration of Triplet Loss with CLIP for improved embedding learning

**Contents**:

- Hard triplet mining with dynamic sample selection
- Combined loss (Triplet + Cross-Entropy) with configurable weighting
- Embedding space optimization for better class separation
- Comprehensive training visualization

**Key Features**:

- **Triplet Loss**: Margin-based loss for embedding space optimization
- **Hard Mining**: Automatic selection of hardest positive and negative pairs
- **Loss Weighting**: λ=0.3 for triplet loss contribution
- **Visualization**: Training curves and loss breakdown analysis

### 3. clip_vpt.ipynb

**Purpose**: Visual Prompt Tuning for parameter-efficient CLIP adaptation

**Contents**:

- Deep VPT implementation with layer-wise prompt injection
- Shared and layer-specific prompt configurations
- Comparison with baseline CLIP models
- Comprehensive performance analysis

**Key Features**:

- **Parameter Efficiency**: Only prompts and classification head are trainable
- **Deep Injection**: Prompts injected at each transformer layer
- **Flexible Configuration**: Support for shared and layer-specific prompts
- **Comparative Analysis**: Direct comparison with vanilla CLIP and LoRA variants

## Results from Notebook Implementations

### MixUp Augmentation Results

- **Training**: Stable convergence with gradual loss reduction
- **Validation**: Achieved 90.88% accuracy with smooth training dynamics
- **Test Performance**: **91.09%** accuracy with improved generalization
- **Key Insight**: MixUp provided smoother training curves and reduced overfitting

### Triplet Loss Results

- **Training**: Combined loss optimization with triplet component ~0.05
- **Validation**: Reached 90.98% accuracy with embedding space regularization
- **Test Performance**: **90.93%** accuracy
- **Key Insight**: Triplet loss provided modest regularization but limited performance gains over baseline

### Visual Prompt Tuning Results

- **Training**: Rapid convergence with training accuracy reaching 100%
- **Validation**: Achieved 93.63% accuracy before early stopping
- **Test Performance**: **87.98%** accuracy (indicated potential overfitting)
- **Key Insight**: VPT showed strong training performance but reduced generalization on test set

## Comparative Performance

| Technique        | Validation Accuracy | Test Accuracy | Key Characteristic       |
| ---------------- | ------------------- | ------------- | ------------------------ |
| **MixUp**        | 90.88%              | **91.09%**    | Best generalization      |
| **Triplet Loss** | 90.98%              | 90.93%        | Embedding regularization |
| **VPT**          | 93.63%              | 87.98%        | Training efficiency      |

## Technical Implementation Details

### MixUp Configuration

- Mixing coefficient: α=0.2 (Beta distribution)
- Linear interpolation of image pixels and labels
- Modified cross-entropy loss for soft targets

### Triplet Loss Configuration

- Margin: 1.0
- Triplet weight: λ=0.3
- Hard negative mining within batches

### VPT Configuration

- Prompt size: 16 tokens
- Deep injection across all transformer layers
- Shared prompts across layers
- Learning rate: 5e-3 for prompt parameters

## Notes

- All experiments use consistent CLIP backbone (ViT-B/32)
- Early stopping employed to prevent overfitting
- Mixed precision training used where supported
- Random seed 42 ensures reproducibility
- Models evaluated on Oxford Flowers-102 test set (6,149 images)

## Requirements

Key dependencies:

- PyTorch
- Transformers
- TorchVision
- scikit-learn (for metrics)
- Matplotlib, Seaborn (for visualization)
- tqdm (for progress bars)

## Execution

Run notebooks independently:

```bash
jupyter notebook clip_mixup.ipynb
jupyter notebook clip_triplet_loss.ipynb
jupyter notebook clip_vpt.ipynb
```

Each notebook contains complete self-contained implementations with data loading, model training, and evaluation.

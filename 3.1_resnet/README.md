# Flower Recognition with ResNet Documentation

## Notebooks and Their Functions

### 1. Base ResNet Implementation

**File**: `ResNet/ResNet18_Standard/base_resnet.ipynb`

- **Purpose**: Standard ResNet-18 with full fine-tuning
- **Output**:
  - Model: `best_resnet18_original_params.pt`
  - Training curves: `ResNet18_Original_training_curves.png`
  - Logs: `resnet18_original.txt`
- **Performance**: 88.03% test accuracy

### 2. ResNet-18 with Frozen Stages

**File**: `ResNet/ResNet18_Frozen/resnet18_3frozen_stages.ipynb`

- **Purpose**: ResNet-18 with first 3 stages frozen (transfer learning)
- **Output**:
  - Model: `best_resnet18_frozen_params.pt`
  - Training curves: `ResNet18_Frozen_training_curves.png`
  - Logs: `resnet18_frozen.txt`
- **Performance**: 88.68% test accuracy

### 3. ResNet-34 with Frozen Stages

**File**: `ResNet/ResNet34_Frozen/resnet34_with_3frozen_stages.ipynb`

- **Purpose**: Deeper ResNet-34 with first 3 stages frozen
- **Output**:
  - Model: `best_resnet34_frozen_params.pt`
  - Training curves: `ResNet34_Frozen_training_curves.png`
  - Logs: `resnet34_frozen.txt`
- **Performance**: 90.00% test accuracy

### 4. Experiment Comparison

**File**: `ResNet/experiment_results_comparison.ipynb`

- **Purpose**: Comparative analysis of all ResNet experiments
- **Output**: `all_experiments_comparison.png`

### 5. Few-Shot Learning Analysis

**File**: `ResNet/ResNet34_frozen_few_shot_analysis/resnet34_few_shot.ipynb`

- **Purpose**: Data efficiency analysis with limited training samples
- **Output**:
  - Analysis plot: `resnet34_few_shot_analysis.png`
  - Results CSV: `resnet34_few_shot_results.csv`
- **Data fractions tested**: 10%, 20%, 30%, 50%, 75%, 100%

## Model Checkpoints Location

All trained models are saved in their respective experiment folders:

- `ResNet/ResNet18_Standard/model_checkpoints/`
- `ResNet/ResNet18_Frozen/model_checkpoints/`
- `ResNet/ResNet34_Frozen/model_checkpoints/`

## Performance Summary

| Model                | Test Accuracy | Key Feature          |
| -------------------- | ------------- | -------------------- |
| ResNet-18 Standard   | 88.03%        | Full fine-tuning     |
| ResNet-18 Frozen     | 88.68%        | Transfer learning    |
| **ResNet-34 Frozen** | **90.00%**    | **Best performance** |

## Running the Notebooks

### Prerequisites

```bash
pip install torch torchvision pandas numpy matplotlib scikit-learn pillow scipy seaborn tqdm
```

### Execution Order

1. Start with `base_resnet.ipynb` for baseline
2. Run frozen variants (`resnet18_3frozen_stages.ipynb`, `resnet34_with_3frozen_stages.ipynb`)
3. Execute few-shot analysis (`resnet34_few_shot.ipynb`)
4. Run comparison analysis (`experiment_results_comparison.ipynb`)

### Expected Outputs for Each Notebook

- **Training logs** (`.txt` files) with epoch-wise metrics
- **Model checkpoints** (`.pt` files) for best performing models
- **Visualization plots** (`.png` files) of training curves
- **Performance summaries** printed in notebook outputs

## Key Findings

1. **Architecture Depth**: ResNet-34 outperforms ResNet-18 (90.00% vs 88.68%)
2. **Transfer Learning**: Frozen stages strategy improves performance over full fine-tuning
3. **Data Efficiency**: ResNet-34 achieves 74% of full performance with only 50% training data
4. **Training Time**: ResNet-34 requires ~75% more training time than ResNet-18

## File Structure

```
SC4001flowers/
├── ResNet/
│   ├── ResNet18_Standard/
│   ├── ResNet18_Frozen/
│   ├── ResNet34_Frozen/
│   ├── ResNet34_frozen_few_shot_analysis/
│   ├── experiment_results_comparison.ipynb
│   └── model_checkpoints/
└── data/
    └── flowers-102/
```

## Notes

- All experiments use consistent preprocessing and data augmentation
- Random seed 42 ensures reproducibility
- Early stopping and learning rate scheduling employed across all experiments

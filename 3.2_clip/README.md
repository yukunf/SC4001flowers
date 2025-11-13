# Flower Recognition with CLIP Documentation

This folder contains the implementation for fine-grained flower classification on the Oxford Flowers-102 dataset using CLIP with parameter-efficient fine-tuning techniques.

## File Structure

```
3.2_clip/
├── clip_prep.py          # Dataset preprocessing and caching
├── clip_train.py         # Main training script for CLIP + LoRA
├── clip_notebook.ipynb   # Interactive notebook with full pipeline
├── clip_label_acc.png    # Performance visualization
```

## Script Descriptions

### 1. clip_prep.py

**Purpose**: Data preprocessing and caching for efficient training

**Functions**:

- Downloads and preprocesses Oxford Flowers-102 dataset
- Applies CLIP processor to images and caches tensors
- Creates memory-mapped datasets for fast loading
- Handles train/val/test splits according to official dataset split

**Key Features**:

- Automatic dataset download and caching
- Support for FP16 preprocessing to save memory
- Garbage collection and memory optimization

### 2. clip_train.py

**Purpose**: Main training script for CLIP with LoRA fine-tuning

**Components**:

- **LoRA Injection**: Injects low-rank adapters into CLIP's attention layers
- **Dual Loss Training**: Combines classification loss with text-image alignment loss
- **Early Stopping**: Prevents overfitting with configurable patience
- **Mixed Precision**: Uses AMP for faster training on compatible hardware

**Training Strategy**:

- Freezes CLIP backbone parameters
- Trains only LoRA adapters and linear classification head
- Uses weighted combination of classification loss and text alignment loss

### 3. clip_notebook.ipynb

**Purpose**: Interactive exploration and experimentation

**Contents**:

- Complete training pipeline with visualizations
- Visual Prompt Tuning (VPT) implementation
- Model saving/loading utilities
- Performance evaluation and testing

## Execution Order

1. **First**: Run `clip_prep.py` to preprocess and cache the dataset

   ```bash
   python clip_prep.py
   ```

2. **Then**: Execute training with `clip_train.py`

   ```bash
   python clip_train.py
   ```

3. **Alternatively**: Use the notebook for interactive experimentation
   ```bash
   jupyter notebook clip_notebook.ipynb
   ```

## Results from Notebook Implementation

The notebook implementation achieved the following results:

**CLIP + LoRA Training**:

- **Validation Accuracy**: 91.76% (epoch 1)
- **Text Alignment Accuracy**: 78.53% (epoch 1)
- Training demonstrated stable convergence before manual interruption for development efficiency

**Visual Prompt Tuning (VPT) Experiment**:

- **Test Accuracy**: 90.86%
- **Text Alignment Accuracy**: 77.13%
- Early stopping triggered at epoch 5, indicating efficient convergence
- VPT showed slightly lower performance compared to standard LoRA fine-tuning

## Configuration

The training uses the following key hyperparameters:

- **Model**: `openai/clip-vit-base-patch32`
- **Batch Size**: 32
- **Epochs**: 40
- **LoRA Rank**: 8
- **Learning Rates**:
  - Head: 1e-3
  - LoRA: 1e-4
- **Text Loss Weight**: 0.3
- **Early Stopping**: Enabled with patience of 3 epochs

## Notes

- All experiments use consistent preprocessing with random seed 42 for reproducibility
- Early stopping and mixed precision training employed
- Dataset follows official Oxford Flowers-102 split: 1,020 train, 1,020 validation, 6,149 test images
- Model checkpoints save only trainable parameters (LoRA adapters + classification head) for efficiency

## Requirements

Key dependencies:

- PyTorch
- Transformers
- TorchVision
- tqdm
- CUDA-compatible GPU recommended

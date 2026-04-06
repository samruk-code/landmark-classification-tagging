# Landmark Classification & Tagging for Social Media

A deep learning project that automatically identifies and tags landmarks in photos — the kind of feature used in social media apps to label images with "Eiffel Tower" or "Machu Picchu" without any user input.

---

## Project Description

Photo-sharing platforms often struggle to provide location context for images uploaded without GPS metadata. This project tackles that problem by training a convolutional neural network to recognize landmarks directly from pixel data.

Two approaches are implemented and compared:
1. **A custom CNN built from scratch** using residual blocks
2. **Transfer learning** using a pretrained ResNet backbone

The final model is wrapped in a TorchScript-compatible inference pipeline, making it ready for deployment.

---

## Problem Statement

Given an image that may contain a well-known landmark, the goal is to predict which landmark appears in the image. The model must generalize across varying lighting conditions, camera angles, and image compositions.

---

## Problem-Solving Method

### Data Pipeline (`src/data.py`)

- Dataset: `landmark_images/` with `train/` and `test/` subdirectories in PyTorch `ImageFolder` format
- The validation set is a random 20% split of the training data (drawn from `train/` using `SubsetRandomSampler`), not a separate folder
- Training transforms: `Resize(256)` → `RandomResizedCrop(224, scale=0.8–1.0)` → `RandAugment(num_ops=2, magnitude=9)` → `RandomHorizontalFlip` → `ToTensor` → `Normalize`
- Validation/test transforms: `Resize(256)` → `CenterCrop(224)` → `ToTensor` → `Normalize`
- Per-channel mean and standard deviation are computed over the full dataset once and cached to `mean_and_std.pt`

### Approach 1 — CNN from Scratch (`src/model.py`)

A custom `MyModel` class is built with a ResNet-inspired architecture.

**ResidualBlock** (core building unit):
- Two sequential `Conv2d(3×3)` layers, each followed by `BatchNorm2d` and `ReLU`
- Skip connection: identity if input/output dimensions match; otherwise a `Conv2d(1×1)` + `BatchNorm2d` projection
- Output: element-wise sum of the residual and skip, passed through a final `ReLU`

**MyModel** stacks 4 residual blocks:

| Block | In Channels | Out Channels | Stride |
|-------|-------------|--------------|--------|
| 1     | 3           | 64           | 1      |
| 2     | 64          | 128          | 2      |
| 3     | 128         | 256          | 2      |
| 4     | 256         | 512          | 2      |

After the blocks: `AdaptiveAvgPool2d(1×1)` → `Flatten` → fully connected head:

`Linear(512→512)` → `BatchNorm1d` → `ReLU` → `Dropout(0.7)` → `Linear(512→192)` → `BatchNorm1d` → `ReLU` → `Dropout(0.7)` → `Linear(192→num_classes)`

### Approach 2 — Transfer Learning (`src/transfer.py`)

- Loads a pretrained `resnet18` from `torchvision.models`
- Freezes all backbone parameters (`requires_grad = False`)
- Replaces the final `fc` layer with `Linear(num_features, n_classes)`
- Only the new classification head is trained

### Training Loop (`src/train.py` / `src/optimization.py`)

- **Loss:** `CrossEntropyLoss` with label smoothing of 0.1
- **Optimizer:** SGD (default) or Adam
  - SGD defaults: `lr=0.01`, `momentum=0.5`, `weight_decay=0`
  - Adam uses `lr` and `weight_decay` only (momentum is not applicable)
- **LR Scheduler:** `ReduceLROnPlateau(mode='min')` — reduces learning rate when validation loss stops improving
- **Checkpointing:** model weights are saved when validation loss decreases by more than 1% relative to the current best

### Inference (`src/predictor.py`)

The `Predictor` class wraps any trained model using `nn.Sequential` (instead of `transforms.Compose`) for full TorchScript compatibility. Its internal transform pipeline is:

`Resize(256)` → `CenterCrop(224)` → `ConvertImageDtype(float)` → `Normalize`

It accepts an image tensor (integer or float dtype) and outputs a softmax probability vector over all landmark classes.

---

## Project Structure

```
landmark-classification-tagging/
├── src/
│   ├── data.py               # Data loading and augmentation
│   ├── model.py              # Custom CNN (ResNet-inspired, from scratch)
│   ├── transfer.py           # Transfer learning with pretrained ResNet
│   ├── train.py              # Training, validation, and test loops
│   ├── optimization.py       # Loss function and optimizer factory
│   ├── predictor.py          # TorchScript-compatible inference wrapper
│   ├── helpers.py            # Dataset download, normalization, plotting
│   └── create_submit_pkg.py  # Packages notebooks and source for submission
├── cnn_from_scratch.ipynb    # Notebook: train and evaluate custom CNN
├── transfer_learning.ipynb   # Notebook: train and evaluate transfer learning model
├── app.ipynb                 # Notebook: end-to-end demo application
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.11.0
- torchvision 0.12.0

Install all dependencies:

```bash
pip install -r requirements.txt
```

### Dataset

The dataset is downloaded automatically on first run via `src/helpers.py`. Alternatively, place the `landmark_images/` folder in the project root manually.

### Running the Notebooks

Open the notebooks in order:

1. `cnn_from_scratch.ipynb` — trains and evaluates the custom CNN
2. `transfer_learning.ipynb` — trains and evaluates the transfer learning model
3. `app.ipynb` — runs the end-to-end application demo

### Running Tests

Each source file contains its own pytest suite:

```bash
pytest src/
```

---

## Conclusion

Transfer learning with a pretrained ResNet significantly outperformed the custom CNN in both accuracy and training speed, which is expected given the limited dataset size. The custom ResNet-inspired model still demonstrated solid performance, validating the residual block design. The final inference pipeline is production-ready, supporting TorchScript export for deployment in mobile or server environments.

# MNIST Convolutional Neural Network with Boat Framework

This example demonstrates how to build, train, and evaluate a convolutional neural network for MNIST digit recognition using the Boat deep learning framework.

## Overview

The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). Each image is 28x28 pixels grayscale.

## Model Architecture

A simple yet effective CNN architecture for MNIST:

```
Input: 1x28x28 (channels x height x width)
├── Conv2D(32, kernel_size=3x3, padding=1)
├── ReLU()
├── MaxPool2D(kernel_size=2x2, stride=2)
├── Conv2D(64, kernel_size=3x3, padding=1)
├── ReLU()
├── MaxPool2D(kernel_size=2x2, stride=2)
├── Flatten()
├── Dense(128)
├── ReLU()
├── Dense(10)
└── Softmax()
```

## Implementation Plan

### 1. Data Preparation
- Use Python `mnist` package to load MNIST data
- Convert to Boat tensor format (FLOAT32)
- Normalize pixel values to [0, 1] range
- Save as binary files for C consumption

### 2. Model Definition
- Create layers using Boat API:
  - `boat_conv_layer_create()`
  - `boat_pool_layer_create()`
  - `boat_dense_layer_create()`
  - `boat_relu_layer_create()`
  - `boat_softmax_layer_create()`
  - `boat_flatten_layer_create()`

### 3. Training Loop
- Forward pass through all layers
- Compute cross-entropy loss
- Backward pass (gradient computation)
- Update parameters using Adam optimizer
- Track training loss and accuracy

### 4. Evaluation
- Forward pass on test set
- Compute accuracy
- Display sample predictions

### 5. Performance Targets
- Training accuracy: >95%
- Test accuracy: >95%
- Reasonable training time (minutes, not hours)

## Files

- `mnist.c`: Main C implementation
- `mnist_data.py`: Python script to prepare MNIST data
- `train_images.bin`, `train_labels.bin`: Training data
- `test_images.bin`, `test_labels.bin`: Test data
- `CMakeLists.txt`: Build configuration
- `README.md`: Usage instructions

## Dependencies

- Boat framework (built from parent directory)
- Python 3.x with `mnist` package for data preparation
- CMake 3.10+

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Running

1. Prepare data:
```bash
python mnist_data.py
```

2. Train and evaluate:
```bash
./mnist
```

## Implementation Notes

- Uses float32 precision for all computations
- Implements mini-batch training for efficiency
- Includes validation during training
- Saves model checkpoints (optional)
- Displays progress metrics

This example serves as a practical demonstration of Boat's capabilities for computer vision tasks and provides a template for building other CNN-based models.
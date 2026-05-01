#!/usr/bin/env python3
"""
CIFAR-10 Data Preparation for Boat Framework

Downloads and prepares CIFAR-10 data for C consumption.
Saves as binary files matching the MNIST example format.
"""

import struct
import os
import sys


def save_tensor_binary(filename, data, dtype='float32'):
    """Save tensor data as binary file (same format as MNIST example)."""
    if dtype == 'float32':
        data = data.astype('float32')
    elif dtype == 'uint8':
        data = data.astype('uint8')
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    with open(filename, 'wb') as f:
        # Write shape dimensions
        f.write(struct.pack('I', len(data.shape)))
        for dim in data.shape:
            f.write(struct.pack('I', dim))
        # Write data
        f.write(data.tobytes())

    print(f"  Saved {filename} with shape {data.shape}")


def load_cifar10():
    """Load CIFAR-10 data using best available backend."""
    # Force dummy data if environment variable set
    if os.environ.get('USE_DUMMY_DATA'):
        print("Using dummy data (forced by USE_DUMMY_DATA)")
        import numpy as np
        n_train = 100
        n_test = 20
        train_images = np.random.randint(0, 256, size=(n_train, 32, 32, 3), dtype=np.uint8)
        train_labels = np.random.randint(0, 10, size=(n_train,), dtype=np.uint8)
        test_images = np.random.randint(0, 256, size=(n_test, 32, 32, 3), dtype=np.uint8)
        test_labels = np.random.randint(0, 10, size=(n_test,), dtype=np.uint8)
        return train_images, train_labels, test_images, test_labels

    # Try tensorflow.keras first
    try:
        import tensorflow as tf
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
        train_labels = train_labels.flatten()
        test_labels = test_labels.flatten()
        print("Loaded CIFAR-10 via tensorflow.keras")
        return train_images, train_labels, test_images, test_labels
    except ImportError:
        pass

    # Try torchvision second
    try:
        import torchvision
        import torch
        import numpy as np
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data_raw', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data_raw', train=False, download=True, transform=transform)

        # Convert to numpy (ToTensor gives CHW in [0,1], scale back to uint8 range)
        train_images = np.stack([(img.numpy() * 255).transpose(1, 2, 0) for img, _ in trainset]).astype(np.uint8)
        train_labels = np.array([label for _, label in trainset])
        test_images = np.stack([(img.numpy() * 255).transpose(1, 2, 0) for img, _ in testset]).astype(np.uint8)
        test_labels = np.array([label for _, label in testset])

        print("Loaded CIFAR-10 via torchvision")
        return train_images, train_labels, test_images, test_labels
    except ImportError:
        pass

    # Generate dummy data as fallback
    print("No ML framework found. Generating dummy data for testing...")
    import numpy as np
    n_train = 100
    n_test = 20
    train_images = np.random.randint(0, 256, size=(n_train, 32, 32, 3), dtype=np.uint8)
    train_labels = np.random.randint(0, 10, size=(n_train,), dtype=np.uint8)
    test_images = np.random.randint(0, 256, size=(n_test, 32, 32, 3), dtype=np.uint8)
    test_labels = np.random.randint(0, 10, size=(n_test,), dtype=np.uint8)
    print(f"Generated dummy data: {n_train} training, {n_test} test samples")
    return train_images, train_labels, test_images, test_labels


def main():
    print("Loading CIFAR-10 data...")

    train_images, train_labels, test_images, test_labels = load_cifar10()

    print(f"Training images shape: {train_images.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")

    # Normalize to [0, 1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Convert from HWC to NCHW: (N, H, W, C) -> (N, C, H, W)
    train_images = train_images.transpose(0, 3, 1, 2)
    test_images = test_images.transpose(0, 3, 1, 2)

    print(f"Reshaped training images: {train_images.shape}")
    print(f"Reshaped test images: {test_images.shape}")

    # Create output directory
    os.makedirs("data", exist_ok=True)

    # Save full datasets
    save_tensor_binary("data/train_images.bin", train_images, 'float32')
    save_tensor_binary("data/train_labels.bin", train_labels, 'uint8')
    save_tensor_binary("data/test_images.bin", test_images, 'float32')
    save_tensor_binary("data/test_labels.bin", test_labels, 'uint8')

    # Save small subsets for quick testing
    save_tensor_binary("data/train_images_small.bin", train_images[:1000], 'float32')
    save_tensor_binary("data/train_labels_small.bin", train_labels[:1000], 'uint8')
    save_tensor_binary("data/test_images_small.bin", test_images[:200], 'float32')
    save_tensor_binary("data/test_labels_small.bin", test_labels[:200], 'uint8')

    print("\nData preparation complete!")
    print("Files saved in 'data/' directory:")
    print("  train_images.bin     - 50,000 training images")
    print("  train_labels.bin     - 50,000 training labels")
    print("  test_images.bin      - 10,000 test images")
    print("  test_labels.bin      - 10,000 test labels")
    print("  *_small.bin          - Subsets for quick testing")


if __name__ == "__main__":
    main()

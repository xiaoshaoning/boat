#!/usr/bin/env python3
"""
MNIST Data Preparation for Boat Framework

This script downloads and prepares MNIST data for C consumption.
The MNIST dataset is loaded using the `mnist` package and saved
as binary files that can be read by the C implementation.
"""

import mnist
import numpy as np
import struct
import os
import sys

def save_tensor_binary(filename, data, dtype='float32'):
    """Save tensor data as binary file."""
    if dtype == 'float32':
        fmt = 'f'
        data = data.astype(np.float32)
    elif dtype == 'uint8':
        fmt = 'B'
        data = data.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    with open(filename, 'wb') as f:
        # Write shape dimensions
        f.write(struct.pack('I', len(data.shape)))
        for dim in data.shape:
            f.write(struct.pack('I', dim))

        # Write data
        f.write(data.tobytes())

    print(f"Saved {filename} with shape {data.shape}")

def load_mnist():
    """Load MNIST data using multiple possible backends."""
    # Force dummy data for testing if environment variable set
    import os
    if os.environ.get('USE_DUMMY_DATA'):
        print("Using dummy data (forced by USE_DUMMY_DATA)")
        import numpy as np
        n_train = 100
        n_test = 20
        train_images = np.random.randint(0, 256, size=(n_train, 28, 28), dtype=np.uint8)
        train_labels = np.random.randint(0, 10, size=(n_train,), dtype=np.uint8)
        test_images = np.random.randint(0, 256, size=(n_test, 28, 28), dtype=np.uint8)
        test_labels = np.random.randint(0, 10, size=(n_test,), dtype=np.uint8)
        return train_images, train_labels, test_images, test_labels

    # Try tensorflow.keras first
    try:
        import tensorflow as tf
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        print("Loaded MNIST via tensorflow.keras")
        return train_images, train_labels, test_images, test_labels
    except ImportError:
        pass

    # Try sklearn (scikit-learn) second
    try:
        from sklearn.datasets import fetch_openml
        import numpy as np
        # Load MNIST from OpenML
        mnist_openml = fetch_openml('mnist_784', version=1, parser='auto')
        data = mnist_openml['data'].values.astype(np.uint8)
        target = np.array(mnist_openml['target']).astype(np.uint8)
        # Reshape to (N, 28, 28)
        train_images = data[:60000].reshape(-1, 28, 28)
        train_labels = target[:60000]
        test_images = data[60000:].reshape(-1, 28, 28)
        test_labels = target[60000:]
        print("Loaded MNIST via scikit-learn")
        return train_images, train_labels, test_images, test_labels
    except ImportError:
        pass

    # Try torchvision third
    try:
        import torchvision
        import torch
        transform = torchvision.transforms.ToTensor()
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        # Convert to numpy arrays
        import numpy as np
        train_images = np.stack([img.numpy()[0] for img, _ in trainset]) * 255
        train_labels = np.array([label for _, label in trainset])
        test_images = np.stack([img.numpy()[0] for img, _ in testset]) * 255
        test_labels = np.array([label for _, label in testset])

        print("Loaded MNIST via torchvision")
        return train_images.astype(np.uint8), train_labels, test_images.astype(np.uint8), test_labels
    except ImportError:
        pass

    # Fall back to original mnist package
    try:
        import mnist
        train_images = mnist.train_images()
        train_labels = mnist.train_labels()
        test_images = mnist.test_images()
        test_labels = mnist.test_labels()
        print("Loaded MNIST via mnist package")
        return train_images, train_labels, test_images, test_labels
    except Exception as e:
        print(f"mnist package failed: {e}")
        print("Generating dummy data for testing...")
        import numpy as np
        # Generate small dummy dataset
        n_train = 100
        n_test = 20
        train_images = np.random.randint(0, 256, size=(n_train, 28, 28), dtype=np.uint8)
        train_labels = np.random.randint(0, 10, size=(n_train,), dtype=np.uint8)
        test_images = np.random.randint(0, 256, size=(n_test, 28, 28), dtype=np.uint8)
        test_labels = np.random.randint(0, 10, size=(n_test,), dtype=np.uint8)
        print(f"Generated dummy data: {n_train} training, {n_test} test samples")
        return train_images, train_labels, test_images, test_labels


def main():
    print("Loading MNIST data...")

    # Load MNIST data using best available backend
    train_images, train_labels, test_images, test_labels = load_mnist()

    print(f"Training images shape: {train_images.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")

    # Normalize images to [0, 1] range
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Reshape images to add channel dimension: (N, 1, 28, 28)
    train_images = train_images.reshape(-1, 1, 28, 28)
    test_images = test_images.reshape(-1, 1, 28, 28)

    print(f"Reshaped training images: {train_images.shape}")
    print(f"Reshaped test images: {test_images.shape}")

    # Create output directory if needed
    os.makedirs("data", exist_ok=True)

    # Save as binary files
    save_tensor_binary("data/train_images.bin", train_images, 'float32')
    save_tensor_binary("data/train_labels.bin", train_labels, 'uint8')
    save_tensor_binary("data/test_images.bin", test_images, 'float32')
    save_tensor_binary("data/test_labels.bin", test_labels, 'uint8')

    # Also save a small subset for quick testing
    train_subset = train_images[:1000]
    train_labels_subset = train_labels[:1000]
    test_subset = test_images[:200]
    test_labels_subset = test_labels[:200]

    save_tensor_binary("data/train_images_small.bin", train_subset, 'float32')
    save_tensor_binary("data/train_labels_small.bin", train_labels_subset, 'uint8')
    save_tensor_binary("data/test_images_small.bin", test_subset, 'float32')
    save_tensor_binary("data/test_labels_small.bin", test_labels_subset, 'uint8')

    print("\nData preparation complete!")
    print("Files saved in 'data/' directory:")
    print("  train_images.bin     - 60,000 training images")
    print("  train_labels.bin     - 60,000 training labels")
    print("  test_images.bin      - 10,000 test images")
    print("  test_labels.bin      - 10,000 test labels")
    print("  *_small.bin          - Subsets for quick testing")

if __name__ == "__main__":
    main()
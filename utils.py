"""
Utility functions for data loading and preprocessing.
"""

import numpy as np
import scipy.io
import struct
import os


def lire_alpha_digit(characters, filepath='binaryalphadigs.mat'):
    """
    Load Binary AlphaDigits data for the specified characters.
    """
    mat = scipy.io.loadmat(filepath)
    dat = mat['dat']

    # Build mapping from character to index
    char_to_idx = {}
    for i in range(10):
        char_to_idx[str(i)] = i
    for i in range(26):
        char_to_idx[chr(ord('A') + i)] = 10 + i

    data_list = []
    for c in characters:
        if isinstance(c, (int, np.integer)):
            idx = int(c)
        elif isinstance(c, str):
            idx = char_to_idx[c.upper()]
        else:
            raise ValueError(f"Invalid character specification: {c}")

        if idx < 0 or idx > 35:
            raise ValueError(f"Character index {idx} out of range [0, 35]")

        # Collect all 39 samples for this character
        for j in range(dat.shape[1]):
            img = dat[idx, j]  # 20x16 binary array
            data_list.append(img.flatten().astype(np.float64))

    data = np.array(data_list)
    return data


def load_mnist(data_dir='mnist_data'):
    """
    Load MNIST dataset from idx files and binarize images.
    """
    X_train = _read_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    y_train = _read_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    X_test = _read_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    y_test = _read_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))

    # Binarize: pixel < 127 -> 0, pixel >= 127 -> 1
    X_train = (X_train >= 127).astype(np.float64)
    X_test = (X_test >= 127).astype(np.float64)

    # One-hot encode labels
    Y_train = np.zeros((y_train.shape[0], 10), dtype=np.float64)
    Y_train[np.arange(y_train.shape[0]), y_train] = 1.0

    Y_test = np.zeros((y_test.shape[0], 10), dtype=np.float64)
    Y_test[np.arange(y_test.shape[0]), y_test] = 1.0

    return X_train, Y_train, X_test, Y_test


def _read_mnist_images(filepath):
    """Read MNIST image file in idx format."""
    with open(filepath, 'rb') as f:
        magic, n_images, n_rows, n_cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"Invalid magic number {magic} for images"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(n_images, n_rows * n_cols).astype(np.float64)
    return data


def _read_mnist_labels(filepath):
    """Read MNIST label file in idx format."""
    with open(filepath, 'rb') as f:
        magic, n_labels = struct.unpack('>II', f.read(8))
        assert magic == 2049, f"Invalid magic number {magic} for labels"
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

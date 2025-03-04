import numpy as np
import pickle
import struct

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)
        return images.astype(np.float32) / 255.0  # Normalize pixel values

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

# Load dataset
x_train = load_mnist_images("mnist/train-images-idx3-ubyte")
y_train = load_mnist_labels("mnist/train-labels-idx1-ubyte")
x_test = load_mnist_images("mnist/t10k-images-idx3-ubyte")
y_test = load_mnist_labels("mnist/t10k-labels-idx1-ubyte")

# Save as pickle file
mnist_data = (x_train, y_train, x_test, y_test)
with open("mnist.pkl", "wb") as f:
    pickle.dump(mnist_data, f)

print("mnist.pkl created successfully!")

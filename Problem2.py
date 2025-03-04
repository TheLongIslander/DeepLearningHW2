import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Load MNIST dataset from mnist.pkl
with open("mnist.pkl", "rb") as f:
    mnist_data = pickle.load(f)

# Unpack dataset
x_train, y_train, x_test, y_test = mnist_data

# Reduce dataset size for efficiency
num_train = 60000  # Full training set
num_test = 10000   # Full test set

x_train = x_train[:num_train]
y_train = y_train[:num_train]
x_test = x_test[:num_test]
y_test = y_test[:num_test]

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_onehot = encoder.transform(y_test.reshape(-1, 1))  # Transform test labels with the same encoder

# Initialize random weights
num_classes = 10
num_features = x_train.shape[1]
W = np.random.randn(num_features, num_classes) * 0.01  # Small random weights

# Softmax function
def softmax(logits):
    logits = np.array(logits)  # Ensure NumPy array
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# Cross-entropy loss
def cross_entropy_loss(y_pred, y_true):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

# Hyperparameters
learning_rate = 0.1
num_epochs = 100  # Train for 100 iterations

# Training loop using Gradient Descent
for epoch in range(num_epochs):
    logits = x_train @ W
    y_pred = softmax(logits)
    
    # Compute gradient
    grad_W = (1 / num_train) * x_train.T @ (y_pred - y_train_onehot)

    # Update weights
    W -= learning_rate * grad_W

    # Compute loss
    loss = cross_entropy_loss(y_pred, y_train_onehot)
    
    # Evaluate on test set every 10 epochs
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        test_logits = x_test @ W
        test_pred = np.argmax(softmax(test_logits), axis=1)
        test_acc = np.mean(test_pred == np.argmax(y_test_onehot, axis=1))
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Print final accuracy
print(f"Final Test Accuracy: {test_acc:.4f}")

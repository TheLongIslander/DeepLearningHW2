import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def compute_gradients(W, x):
    # Forward Pass
    z = W @ x  # Matrix-vector multiplication
    y = relu(z)  # Apply ReLU
    f = np.linalg.norm(y)**2  # L2 norm squared

    # Backward Pass
    df_dy = 2 * y  # Gradient of L2 norm squared
    dz_dy = relu_derivative(z)  # Derivative of ReLU
    df_dz = df_dy * dz_dy  # Chain rule

    df_dW = np.outer(df_dz, x)  # Gradient w.r.t. W
    df_dx = W.T @ df_dz  # Gradient w.r.t. x

    return f, df_dW, df_dx

# Example usage
np.random.seed(42)
W = np.random.randn(3, 3)  # Random 3x3 matrix
x = np.random.randn(3, 1)  # Random 3x1 vector

f_value, grad_W, grad_x = compute_gradients(W, x)

print("Function value (f):", f_value)
print("Gradient w.r.t W:\n", grad_W)
print("Gradient w.r.t x:\n", grad_x)

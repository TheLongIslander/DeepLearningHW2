import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def compute_gradients(W, x):
    # Forward Pass
    z = W @ x 
    y = relu(z)
    f = np.linalg.norm(y)**2
    # Backward Pass
    df_dy = 2 * y 
    dz_dy = relu_derivative(z) 
    df_dz = df_dy * dz_dy

    df_dW = np.outer(df_dz, x)
    df_dx = W.T @ df_dz

    return f, df_dW, df_dx

# Example usage
np.random.seed(42)
W = np.random.randn(3, 3)
x = np.random.randn(3, 1)

f_value, grad_W, grad_x = compute_gradients(W, x)

print("Function value (f):", f_value)
print("Gradient w.r.t W:\n", grad_W)
print("Gradient w.r.t x:\n", grad_x)

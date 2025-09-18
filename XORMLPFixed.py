import numpy as np

# ---- 1. Dataset ----
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
y = np.array([[0],
              [1],
              [1],
              [0]])

# ---- 2. Random seed ----
np.random.seed(42)

# ---- 3. Network sizes ----
input_size = 2
hidden_size = 3   # increased hidden neurons
output_size = 1

# ---- 4. Initialize weights and biases ----
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# ---- 5. Activation functions ----
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - x**2     # derivative in terms of tanh output

# ---- 6. Training settings ----
learning_rate = 0.1
epochs = 10000

# ---- 7. Training loop ----
for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = tanh(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = tanh(z2)

    # Loss (MSE)
    loss = np.mean((y - a2)**2)

    # Backpropagation
    error_output = y - a2
    d_output = error_output * tanh_derivative(a2)

    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * tanh_derivative(a1)

    # Update weights and biases
    W2 += a1.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    # Print progress
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ---- 8. Final predictions ----
print("\nComparison (True vs Predicted):")
for i in range(len(y)):
    print(f"Input: {X[i]}  True: {y[i][0]}  Predicted: {a2[i][0]:.3f}")

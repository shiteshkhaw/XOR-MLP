import numpy as np

# ---- 1. Dataset ----
# Inputs (X) = 4 rows, each with 2 bits
# Outputs (y) = the XOR results (0,1,1,0)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
y = np.array([[0],
              [1],
              [1],
              [0]])

# ---- 2. Random seed for reproducibility ----
np.random.seed(42)

# ---- 3. Network architecture ----
input_size = 2     # 2 input bits
hidden_size = 2    # 2 hidden neurons
output_size = 1    # 1 output neuron

# ---- 4. Initialize weights and biases ----
# W1 = weights from input -> hidden (2x2)
W1 = np.random.randn(input_size, hidden_size)
# b1 = biases for hidden layer (1x2)
b1 = np.zeros((1, hidden_size))

# W2 = weights from hidden -> output (2x1)
W2 = np.random.randn(hidden_size, output_size)
# b2 = bias for output layer (1x1)
b2 = np.zeros((1, output_size))

# ---- 5. Activation functions ----
def sigmoid(x):
    return 1 / (1 + np.exp(-x))   # squashes values into 0..1

def sigmoid_derivative(x):
    return x * (1 - x)            # derivative using output of sigmoid

# ---- 6. Training settings ----
learning_rate = 0.1
epochs = 50000

# ---- 7. Training loop ----
for epoch in range(epochs):

    # ---- Forward pass ----
    z1 = np.dot(X, W1) + b1        # input -> hidden (linear)
    a1 = sigmoid(z1)               # apply activation to hidden
    z2 = np.dot(a1, W2) + b2       # hidden -> output (linear)
    a2 = sigmoid(z2)               # apply activation to output (prediction)

    # ---- Loss calculation ----
    loss = np.mean((y - a2) ** 2)  # mean squared error

    # ---- Backpropagation ----
    error_output = y - a2                          # output error
    d_output = error_output * sigmoid_derivative(a2)   # output delta

    error_hidden = d_output.dot(W2.T)              # pass error back to hidden
    d_hidden = error_hidden * sigmoid_derivative(a1)   # hidden delta

    # ---- Update weights and biases ----
    W2 += a1.T.dot(d_output) * learning_rate       # update W2
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate  # update b2
    W1 += X.T.dot(d_hidden) * learning_rate        # update W1
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate  # update b1

    # ---- Print progress ----
    if epoch % 5000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ---- 8. Final predictions ----
print("\nFinal predictions:")
print(a2.round(3))   # round to 3 decimals for clarity

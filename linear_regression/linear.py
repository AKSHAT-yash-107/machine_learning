import numpy as np

# Training Data
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

# Hyperparameters
learning_rate = 0.01
epochs = 1000
# Initial parameters
m = 0  # slope
c = 0  # intercept

n = len(X)

for i in range(epochs):

    # Predictions
    Y_pred = m * X + c

    # Calculate gradients
    dm = (2/n) * sum((Y_pred - Y) * X)
    dc = (2/n) * sum(Y_pred - Y)

    # Update m and c
    m = m - learning_rate * dm
    c = c - learning_rate * dc

print("Training Complete")
print("Slope (m):", m)
print("Intercept (c):", c)

# Predict new value
x_new = 6
y_new = m * x_new + c
print("Prediction for x=6:", y_new)

#import numpy as np
#from sklearn.linear_model import LinearRegression
#
#
#
#X = np.column_stack((X1, X2))
#
#
#model = LinearRegression()
#
#
#model.fit(X, Y)
#
#
#print("Intercept:", model.intercept_)
#print("Coefficients:", model.coef_)
#
#
#new = [[80, 24]]    
#print("Prediction for X1=80, X2=24 →", model.predict(new))

#
#import numpy as np
#from sklearn.linear_model import LinearRegression
#
## ---- INPUT ARRAYS ----
#
#X1 = [60, 62, 67, 70, 71, 72, 75,78]
#X2 = [22,25,24,20,15,14,14,11]
#Y  = [140,155,159,179,192,200,212,215]
#
## Convert X1 & X2 into 2D array → shape (n_samples, 2)
#X = np.column_stack((X1, X2))
#
## Create model
#model = LinearRegression()
#
## Train the model
#model.fit(X, Y)
#
## Print regression equation
#print("Intercept (b0):", model.intercept_)
#print("Coefficients (b1, b2):", model.coef_)
#
## Example prediction
#new_X1 = 80
#new_X2 = 24
#new_input = [[new_X1, new_X2]]
#
#prediction = model.predict(new_input)
#print(f"Prediction for X1={new_X1}, X2={new_X2} → {prediction[0]}")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# -----------------------------
#  YOUR DATA
# -----------------------------
X1 = np.array([60, 62, 67, 70, 71, 72, 75, 78])
X2 = np.array([22, 25, 24, 20, 15, 14, 14, 11])
Y  = np.array([140,155,159,179,192,200,212,215])

# Combine features into matrix
X = np.column_stack((X1, X2))

# -----------------------------
#  TRAIN MODEL
# -----------------------------
model = LinearRegression()
model.fit(X, Y)

# Print regression equation
print("\nRegression Equation:")
print(f"Y = {model.intercept_:.3f} + ({model.coef_[0]:.3f})*X1 + ({model.coef_[1]:.3f})*X2")

# -----------------------------
#  3D PLOT
# -----------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of actual data
ax.scatter(X1, X2, Y, color='red', s=50, label="Data Points")

# Create grid for plane
x1_range = np.linspace(X1.min(), X1.max(), 20)
x2_range = np.linspace(X2.min(), X2.max(), 20)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

# Predict Y for grid points
y_pred_grid = model.predict(np.column_stack((x1_grid.ravel(), x2_grid.ravel())))
y_pred_grid = y_pred_grid.reshape(x1_grid.shape)

# Regression plane
ax.plot_surface(x1_grid, x2_grid, y_pred_grid, alpha=0.5, cmap='viridis')

# Labels
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Y")
ax.set_title("Multiple Linear Regression (3D Plot)")

plt.show()

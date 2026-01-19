# 3D visualization of cost surface and gradient-descent path
# Data + settings
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # needed for 3D plotting in some mpl versions

x = np.array([1,2,3,4,5], dtype=float)
y = np.array([5,6,10,13,11], dtype=float)
n = len(x)

# Gradient descent hyperparams (edit if you want)
b0 = 3.0
b1 = 2.0
alpha = 0.1          # learning rate
iterations = 20      # number of GD steps

# Cost function (MSE-style with 1/(2n) factor for nicer surface)
def cost(b0, b1, x, y):
    y_hat = b0 + b1 * x
    return 0.5 * np.mean((y_hat - y)**2)

# Gradient (vectorized)
def gradients(b0, b1, x, y):
    y_hat = b0 + b1 * x
    e = y_hat - y
    db0 = np.mean(e)           # dJ/db0
    db1 = np.mean(e * x)       # dJ/db1
    return db0, db1

# Run gradient descent and record history
b0_hist = [b0]
b1_hist = [b1]
cost_hist = [cost(b0, b1, x, y)]

for it in range(iterations):
    db0, db1 = gradients(b0, b1, x, y)
    b0 = b0 - alpha * db0
    b1 = b1 - alpha * db1
    b0_hist.append(b0)
    b1_hist.append(b1)
    cost_hist.append(cost(b0, b1, x, y))

b0_hist = np.array(b0_hist)
b1_hist = np.array(b1_hist)
cost_hist = np.array(cost_hist)

print("Final params after", iterations, "iters: b0 =", b0, ", b1 =", b1)
print("Final cost:", cost_hist[-1])

# Prepare grid for surface plot
# Choose ranges centered around history with margins
b0_min, b0_max = b0_hist.min() - 1.0, b0_hist.max() + 1.0
b1_min, b1_max = b1_hist.min() - 1.0, b1_hist.max() + 1.0

B0 = np.linspace(b0_min, b0_max, 80)
B1 = np.linspace(b1_min, b1_max, 80)
B0_grid, B1_grid = np.meshgrid(B0, B1)

# Compute cost on grid
Z = np.zeros_like(B0_grid)
for i in range(B0_grid.shape[0]):
    for j in range(B0_grid.shape[1]):
        Z[i,j] = cost(B0_grid[i,j], B1_grid[i,j], x, y)

# Plotting
fig = plt.figure(figsize=(12,6))

# 3D surface + path
ax = fig.add_subplot(1,2,1, projection='3d')
surf = ax.plot_surface(B0_grid, B1_grid, Z, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
ax.set_xlabel('b0')
ax.set_ylabel('b1')
ax.set_zlabel('Cost J(b0,b1)')
ax.set_title('Cost Surface with Gradient Descent Path')
fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)

# plot GD path on surface
ax.plot(b0_hist, b1_hist, cost_hist, color='red', marker='o', markersize=5, linewidth=2, label='GD path')
ax.scatter(b0_hist[0], b1_hist[0], cost_hist[0], color='black', s=50, label='start')
ax.scatter(b0_hist[-1], b1_hist[-1], cost_hist[-1], color='yellow', s=60, edgecolors='k', label='end')
ax.legend()

# Contour projection + path (2D view)
ax2 = fig.add_subplot(1,2,2)
# contour levels
levels = np.logspace(np.log10(Z.min()+1e-8), np.log10(Z.max()+1e-8), 25)
# For nicer contours, use linear spacing too
CS = ax2.contour(B0_grid, B1_grid, Z, levels=25, cmap='viridis')
ax2.clabel(CS, inline=1, fontsize=8)
ax2.set_xlabel('b0')
ax2.set_ylabel('b1')
ax2.set_title('Cost Contours and GD Path (projection)')
# plot path projection
ax2.plot(b0_hist, b1_hist, '-o', color='red', markersize=5)
ax2.scatter(b0_hist[0], b1_hist[0], color='black', s=50, label='start')
ax2.scatter(b0_hist[-1], b1_hist[-1], color='yellow', s=60, edgecolors='k', label='end')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
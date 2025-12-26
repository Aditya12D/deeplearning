import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

# Generate data
X, y = make_blobs(n_samples=200, centers=2, cluster_std=0.6, random_state=42)

# Train perceptron
p = Perceptron()
p.fit(X, y)

# Extract parameters
w = p.coef_[0]
b = p.intercept_[0]

print("Weights:", w)
print("Intercept:", b)

# Scatter plot
# plt.figure()
plt.scatter(X[:,0], X[:,1], c=y)

# Create grid
x1_vals = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200)
x2_vals = np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Implicit line: w1*x1 + w2*x2 + b = 0
Z = w[0]*X1 + w[1]*X2 + b

# Plot decision boundary
plt.contour(X1, X2, Z, levels=[0])

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Perceptron Decision Boundary")
plt.show()

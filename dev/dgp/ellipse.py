import matplotlib.pyplot as plt
import numpy as np

# Parameters for the tilted ellipse
h, k = 0.5, 0.5  # center of the ellipse
a, b = 0.4, 0.2  # semi-major and semi-minor axes
theta = np.pi / 4  # rotation angle in radians (45 degrees)

# Define the grid within [0, 1] x [0, 1]
x = np.linspace(0, 1, 800)
y = np.linspace(0, 1, 800)
X, Y = np.meshgrid(x, y)

# Rotate coordinates
X_rot = (X - h) * np.cos(theta) + (Y - k) * np.sin(theta)
Y_rot = -(X - h) * np.sin(theta) + (Y - k) * np.cos(theta)

# Equation of the rotated ellipse
Z = (X_rot / a) ** 2 + (Y_rot / b) ** 2

# Plot
plt.figure(figsize=(6, 6))
plt.contour(X, Y, Z, levels=[1], colors="blue")
plt.title("Tilted Ellipse Inside [0, 1] x [0, 1]")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.gca().set_aspect("equal")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

import sympy as sp

# Symbols
x, y = sp.symbols("x y")
h, k = 0.5, 0.5  # center
a, b = 0.4, 0.2  # semi-axes
theta = sp.pi / 4  # rotation angle

# Rotation expressions
x_prime = (x - h) * sp.cos(theta) + (y - k) * sp.sin(theta)
y_prime = -(x - h) * sp.sin(theta) + (y - k) * sp.cos(theta)

# Rotated ellipse equation
ellipse_eq = (x_prime / a) ** 2 + (y_prime / b) ** 2 - 1

# Expand into general quadratic form
ellipse_eq_expanded = sp.expand(ellipse_eq)

# Convert to standard quadratic form: Ax^2 + Bxy + Cy^2 + Dx + Ey + F
ellipse_eq_expanded

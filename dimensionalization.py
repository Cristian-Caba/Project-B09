import numpy as np
import pandas as pd
import math 
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import scale_coordinates

# get C_x
C_x = scale_coordinates.C_x

# File paths
x_file = r"Airfoil/x_scaled.txt"
y_file = r"Airfoil/y_scaled.txt"

# Read x and y values
# Load x-values completely
with open(x_file, "r") as fx:
    x_values = [float(line.strip()) for line in fx]

# Load y-values only up to the length of x_values
with open(y_file, "r") as fy:
    y_values = []
    for idx, line in enumerate(fy):
        if idx >= len(x_values):  # Stop reading when reaching the end of x_values
            break
        y_values.append(float(line.strip()))


# Ensure x and y have the same number of points
if len(x_values) != len(y_values):
    raise ValueError("x and y files must have the same number of values.")


# Create cubic spline interpolation
cs = CubicSpline(x_values, y_values)

x_values = np.array(x_values)  # convert list to numpy array
y_values = np.array(y_values)

# Create fine x-grid for plotting
x_fine = np.linspace(x_values.min(), 160, 100)
y_fine = cs = CubicSpline(x_values, y_values)(x_fine)

# calculate arc length to define s
s_fine = np.zeros(x_fine.shape)  # Start with s = 0 at the first point
dx = np.diff(x_fine)
dy = np.diff(y_fine)
ds = np.zeros(dx.shape)

print(dx,dy)

for i in range(len(dx)):
    ds[i] = np.sqrt(dx[i]**2 + dy[i]**2)
    
    #slope[i] = dy[i]/dx[i]
    s_fine[i+1] = s_fine[i] + ds[i]  # Cumulative sum


# Function converting array of s values to x/cx values
def convert_sx(s_grid: np.ndarray) -> np.ndarray: 

    # so that when plotting the s vector plugged in and x/cx vector returned
    x_grid = CubicSpline(x_fine, s_fine)(s_grid)
    if sum(s_grid > 160):
        print("s_grid out of interpolation range")

    return x_grid/C_x

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, 'o', label='Original points')
plt.plot(x_fine, y_fine, '-', label='Cubic spline interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Cubic Spline Interpolation of Airfoil Coordinates')
plt.grid(True)
plt.show()













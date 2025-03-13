import numpy as np
import pandas as pd
import math
#import interp1d 
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import CubicSpline

C_X = 1.2728 #meters
Sweep = math.radians(45)

#Convert global coordinate system chord length C_X to local system C_x_small 
         
C_x = C_X * math.cos(Sweep)  #cos45


# interpolate airfoil coordinates as y(x/cx)



# create s(x/cx) as the arc length of y(x/cx)



# File paths
x_file = r"C:\Users\Vladimir\.vscode\Project-B09\Airfoil\x_scaled.txt"
y_file = r"C:\Users\Vladimir\.vscode\Project-B09\Airfoil\y.txt"
s_file = r"C:\Users\Vladimir\.vscode\Project-B09\Airfoil\s_scaled.txt"

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


# Convert y-values to local coordinates
y_values = [y * C_x for y in y_values]  # Convert y-values to local coordinates



# Ensure x and y have the same number of points
if len(x_values) != len(y_values):
    print(len(x_values), len(y_values))
    raise ValueError("x and y files must have the same number of values.")


# Create cubic spline interpolation
cs = CubicSpline(x_values, y_values)

x_values = np.array(x_values)  # convert list to numpy array
y_values = np.array(y_values)

# Create fine x-grid for plotting
x_fine = np.linspace(x_values.min(), x_values.max(), 500)
y_fine = cs = cs = CubicSpline(x_values, y_values)(x_fine)

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













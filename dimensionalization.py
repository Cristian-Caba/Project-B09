import numpy as np
import pandas as pd
import math
import interp1d 

C_X = 1.2728 #meters
Sweep = math.radians(45)

#Convert global coordinate system chord length C_X to local system C_x_small 
         
C_x = C_X * math.cos(Sweep)  #cos45


# interpolate airfoil coordinates as y(x/cx)



# create s(x/cx) as the arc length of y(x/cx)



# File paths
x_file = r"/Airfoil/x_scaled.txt"
y_file = r"/Airfoil/y.txt"
s_file = r"/Airfoil/s_scaled.txt"

# Read x and y values
with open(x_file, "r") as fx, open(y_file, "r") as fy:
    x_values = [float(line.strip()) for line in fx]
    y_values = [float(line.strip()) for line in fy]

# Ensure x and y have the same number of points
if len(x_values) != len(y_values):
    raise ValueError("x and y files must have the same number of values.")

# Compute cumulative arc length s
s_values = [0.0]  # Start with s = 0 at the first point
for i in range(1, len(x_values)):
    dx = x_values[i] - x_values[i - 1]
    dy = y_values[i] - y_values[i - 1]
    ds = math.sqrt(dx**2 + dy**2)  # Pythagorean theorem
    s_values.append(s_values[-1] + ds)  # Cumulative sum

# Write the s values to s.txt
with open(s_file, "w") as fs:
    for s in s_values:
        fs.write(f"{s:.6f}\n")

print(f"Cumulative distances saved to {s_file}")


# Ensure x and s have the same number of points
if len(x_values) != len(s_values):
    raise ValueError("x and s files must have the same number of values.")

#interpolate S using x/cx as a base:

interp_func = interp1d(x_values, s_values, kind='linear', fill_value="extrapolate")













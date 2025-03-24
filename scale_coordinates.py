import numpy as np
import math 

# Convert global coordinate system chord length C_X to local system C_x
C_X = 1272.8  # milimeters
Sweep = math.radians(45)        
C_x = C_X * math.cos(Sweep)  # cos 45deg

C_X = 1272.8  # milimeters
Sweep = math.radians(45)
C_x = C_X * math.cos(Sweep)

y_unscaled_file = r"Airfoil/y.txt"
with open(y_unscaled_file, "r") as fy:
    y_unscaled = [float(line.strip()) for line in fy]
y_crop = (y_unscaled[:65])
y_rev = list(reversed(y_crop))

y_output = np.array(y_rev)
y_output *= C_x

y_scaled_file = r"Airfoil/y_scaled.txt"
with open(y_scaled_file, "w") as fy:
    for y in y_output:
        fy.write(f"{y:.6f}\n")

x_unscaled_file = r"Airfoil/x.txt"
with open(x_unscaled_file, "r") as fx:
    x_unscaled = [float(line.strip()) for line in fx]
x_crop = (x_unscaled[:65])
x_rev = list(reversed(x_crop))

x_output = np.array(x_rev)
x_output *= C_x

x_scaled_file = r"Airfoil/x_scaled.txt"
with open(x_scaled_file, "w") as fx:
    for x in x_output:
        fx.write(f"{x:.6f}\n")

# OUTPUT IN MILLIMETERS






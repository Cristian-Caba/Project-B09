import numpy as np
import math 

C_X = 1800  # milimeters
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

# OUTPUT IN MILIMETERS






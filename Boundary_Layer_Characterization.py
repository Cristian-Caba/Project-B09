import numpy as np
#from scipy.integrate import trapz
from numpy import trapz

import pandas as pd
from u_infinity import compute_u_inf_for_config_span

# Example data arrays (from your dataset)
"""
y and u depend on x coordinate
x = ???

"""
#y = np.array([y1, y2, y3, ...])  # wall-normal coordinates sorted in increasing order
#u = np.array([u1, u2, u3, ...])  # corresponding u-velocity values at each y


def extract_unique_x_y(filename):
    """
    Reads a file and extracts unique x and y values.
    
    Parameters:
    filename (str): Path to the file containing the data.
    
    Returns:
    unique_x (numpy array): Unique x-coordinates
    unique_y (numpy array): Unique y-coordinates
    """
    x_values = []
    y_values = []
    u_values = []
    v_values = []
    
    with open(filename, 'r') as file:
        next(file)  # Skip first line
        next(file)  # Skip second line
        for line in file:
            line = line.strip()
            if line and not line.startswith(("X", "#", "Title")):  # Skip headers/comments
                data = line.split(',')  # Explicitly split on commas

                print(line) 
                print()
                try:
                    x_values.append(float(data[0]))  # X coordinate
                    y_values.append(float(data[1]))  # Y coordinate
                    u_values.append(float(data[2]))  # U velocity
                    v_values.append(float(data[3]))  # V velocity
    
                except ValueError:
                    print("Skipping line (invalid data): ")
                    print()
    
    unique_x = np.unique(np.array(x_values)) if x_values else np.array([])
    unique_y = np.unique(np.array(y_values)) if y_values else np.array([])
    
    return unique_x, unique_y, u_values, v_values


# Example usage
filename = "PIV_planes/Case_CC_Span_1.txt_u.csv"
unique_x, unique_y, u, v = extract_unique_x_y(filename)

print("Unique X values:", unique_x)
print("Unique Y values:", unique_y)


def calculate_boundary_layer_quantities(y, u, Ue):
    """
    Calculate the displacement thickness (δ*) and momentum thickness (θ)

    Parameters:
    y (numpy array): wall-normal positions (sorted from the surface upwards)
    u (numpy array): streamwise velocity component at positions y
    Ue (float): free stream velocity

    Returns:
    delta_star (float): displacement thickness
    theta (float): momentum thickness
    """

    # Calculate integrand for displacement thickness
    integrand_delta_star = 1 - (u / Ue)
    delta_star = np.trapz(integrand_delta_star, y)

    # Calculate integrand for momentum thickness
    integrand_theta = (u / Ue) * (1 - (u / Ue))
    theta = np.trapz(integrand_theta, y)

    return delta_star, theta

# Cap on max y coordinate for accuracy reasons
y_target = 3.5

#IS THIS DIMENSIONALISED OR NOT?
csv_file = "PIV_planes\Case_CC_Span_1.txt_u.csv"

# Read the data
df = pd.read_csv(csv_file, index_col=0)
# Convert index (Y) and columns (X) to float
df.index = df.index.astype(float)
df.columns = df.columns.astype(float)

# Locate the row in the DataFrame index that is closest to y_target
all_y = df.index.values
idx_closest = np.argmin(np.abs(all_y - y_target))
actual_y = all_y[idx_closest]

print(actual_y)



for span in range(1, 25):
    # Use the helper function to get Ue for each span
    Ue = compute_u_inf_for_config_span("CC", span)
    # Compute the displacement thickness for each span
    delta_star, theta = calculate_boundary_layer_quantities(y, u, Ue)

    print(f"Span {span}: Displacement thickness δ* (CC) = {delta_star:.4f}")


for span in range(1, 25):
    # Use the helper function to get Ue for each span
    Ue = compute_u_inf_for_config_span("SC", span)
    # Compute the displacement thickness for each span
    delta_star, theta = calculate_boundary_layer_quantities(y, u, Ue)

    print(f"Span {span}: Displacement thickness δ* (SC) = {delta_star:.4f}")


# End of Boundary_Layer_Characterization.py
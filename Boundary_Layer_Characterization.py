import numpy as np
#from scipy.integrate import trapezoid
from numpy import trapezoid
from pathlib import Path
import math

import pandas as pd
#from u_infinity import compute_u_inf_for_config_span

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Example function to retrieve free-stream velocity as a function of x.
# Replace with your actual implementation/data source.
# ---------------------------------------------------------------------
def get_u_e(xcoord, filename,filenameu):
    """
    Return free-stream velocity at a given x coordinate.
    In practice, this might come from an external function,
    interpolation from a table, or other data.
    """
    
    minvort = 100000
    rowminvort = 0
    with open(filename) as file:
        matrix = np.genfromtxt(file, delimiter=',', filling_values=np.nan)
        
        for i in range(35,45):
            if matrix[i][xcoord] < minvort:
                minvort=matrix[i][xcoord]
                rowminvort = i
    with open(filenameu) as file:
        matrix = np.genfromtxt(file, delimiter=',', filling_values=np.nan)
        
    return float(matrix[rowminvort+1][xcoord+1])
         

# ---------------------------------------------------------------------
# Function to compute displacement and momentum thickness
# given discrete y, u(y), and a known free-stream velocity u_e.
# ---------------------------------------------------------------------
def compute_boundary_layer_params(y_array, u_array, u_e):
    """
    Computes displacement thickness (delta_star) and momentum thickness (theta)
    from discrete velocity data u_array at coordinates y_array,
    and a free-stream velocity u_e.

    Parameters:
    -----------
    y_array : 1D numpy array of y-coordinates
    u_array : 1D numpy array of velocity at those y-coordinates
    u_e     : scalar (free-stream velocity) or array if needed

    Returns:
    --------
    delta_star : float
    theta      : float
    """
    # Normalize velocity by u_e
    u_ratio = u_array / u_e

    # For displacement thickness: delta* = ∫ [1 - (u/u_e)] dy
    integrand_delta = 1.0 - u_ratio

    # For momentum thickness: theta = ∫ [(u/u_e)*(1 - u/u_e)] dy
    integrand_theta = u_ratio * (1.0 - u_ratio)

    # Perform numerical integration using numpy.trapezoid
    delta_star = np.trapezoid(integrand_delta * (-1), x=y_array)
    theta      = np.trapezoid(integrand_theta * (-1), x=y_array)

    return delta_star, theta

# ---------------------------------------------------------------------
# Main script to read all files, compute BL parameters, and plot.
# ---------------------------------------------------------------------
def main():
    # Directory where your CSV files are located
    data_dir = "./PIV_planes"  # Change to your path if needed
    counter = 0
    # We will store results for each case (i=1..25)
    # Each file can contain multiple x-stations, so we get arrays of x, delta*, theta
    for i in range(1, 25):
        # Filenames for u and v
        file_u = os.path.join(data_dir, f"Case_SC_Span_{i}.txt_u.csv")
        file_v = os.path.join(data_dir, f"Case_SC_Span_{i}.txt_v.csv")
        file_vort = os.path.join( f"./Vorticity/Case_SC_Span_{i}.txt_vorticity.csv")
        # --- READ THE U-COMPONENT CSV ---
        # We assume the CSV is structured such that:
        #   - row 0: [NaN, x1, x2, x3, ...]
        #   - col 0: [NaN, y1, y2, y3, ...]
        #   - interior: velocity values
        # Using pandas, read with header=None so it doesn't treat row0 as column names
        try:
            df_u = pd.read_csv(file_u, header=None)
        except FileNotFoundError:
            print(f"File not found: {file_u}. Skipping.")
            continue

        # Extract x-coordinates from row 0, skipping the very first cell
        x_coords = df_u.iloc[0, 30:300].values.astype(float)
        C_X = 1272.8  # milimeters
        Sweep = math.radians(45)
        C_x = C_X * math.cos(Sweep)
        x_coords = x_coords / C_x  # Convert to local system

        # Extract y-coordinates from col 0, skipping the very first cell
        y_coords = df_u.iloc[1:, 0].values.astype(float)


        # Convert y from mm to m for integration:
        y_coords_m = y_coords / 1000.0

        # Extract the velocity data
        # This should be a 2D array of shape (len(y_coords), len(x_coords))
        u_data = df_u.iloc[1:, 30:300].values.astype(float)

        # Check that shapes match expectations
        ny = len(y_coords)
        nx = len(x_coords)
        if u_data.shape != (ny, nx):
            print("Data shape mismatch! Check CSV formatting.")
            continue

        # We will compute delta*(x) and theta(x) for each column
        delta_star_vals = []
        theta_vals      = []

        # Loop over each x-station (column)
        for ix in range(nx):
            # velocity profile in y at this x
            u_profile = u_data[:, ix]
            x_val = x_coords[ix]
            # Get free-stream velocity at this x
            u_e = get_u_e(ix+30,file_vort,file_u)

            # Compute boundary-layer parameters
            #print()
            #print("VEL PROFILE: ", u_profile)
            #print("Free stream: ", u_e)
            #print()
            dstar, th = compute_boundary_layer_params(y_coords_m, u_profile, u_e)
            dstar = dstar * 1000.0
            th = th * 1000.0
            delta_star_vals.append(dstar)
            theta_vals.append(th)

        # Convert results to numpy arrays for convenience
        delta_star_vals = np.array(delta_star_vals)
        theta_vals      = np.array(theta_vals)

        # --- (Optional) READ THE V-COMPONENT CSV ---
        # If you need to do something with v, do so similarly:
        if os.path.exists(file_v):
            df_v = pd.read_csv(file_v, header=None)
            # parse similarly if needed:
            # v_data = df_v.iloc[1:, 1:].values.astype(float)
            # etc...
        else:
            print(f"No v-file found for i={i}: {file_v}")



    
        # --- PLOTTING ---
        # Plot delta* and theta vs x

        # PROBLEM: APPARENT DATA ANOMALIES AT INTERVAL ENDS !!!!
        # SOLUTION: CROP TO AVOID ANOMALIES                 !!!!

        #mask = (x_coords >= 120) & (x_coords <= 155)
        #x_coords_crop     = x_coords[mask]
        #delta_star_crop   = delta_star_vals[mask]
        #theta_crop        = theta_vals[mask]

        plt.figure(figsize=(8, 6))
        plt.plot(x_coords, delta_star_vals, label=r'$\delta^*$')
        plt.plot(x_coords, theta_vals,      label=r'$\theta$')
        plt.xlabel("x-coordinate")
        plt.ylabel("Thickness (mm)")
        plt.title("Boundary Layer Parameters (x = ? to ?)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        #plt.show()

        # Save in the "plots" folder
        plot_dir = "BL_plots_SC"
        outname = os.path.join(plot_dir, f"BL_Parameters_Case_{i}.png")
        plt.savefig(outname, dpi=150)
        plt.close()
        
        print(f"Processed i={i}: saved plot to {outname}")

if __name__ == "__main__":
    main()

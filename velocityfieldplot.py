import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def block_average(U, V, nx_coarse, ny_coarse):
    """Downsample the velocity field by averaging over blocks."""
    nx, ny = U.shape
    factor_x = nx // nx_coarse
    factor_y = ny // ny_coarse

    # Ensure the dimensions are divisible (crop excess if needed)
    nx_trim = factor_x * nx_coarse
    ny_trim = factor_y * ny_coarse
    U_trimmed = U[:nx_trim, :ny_trim]
    V_trimmed = V[:nx_trim, :ny_trim]

    # Reshape and compute block averages
    U_coarse = U_trimmed.reshape(nx_coarse, factor_x, ny_coarse, factor_y).mean(axis=(1, 3))
    V_coarse = V_trimmed.reshape(nx_coarse, factor_x, ny_coarse, factor_y).mean(axis=(1, 3))

    return U_coarse, V_coarse, factor_x, factor_y

def main():
    folder_path = "PIV_planes"

    # Create a subfolder (images) to save our plots
    save_folder = os.path.join(folder_path, "images")
    os.makedirs(save_folder, exist_ok=True)

    search_patternU = os.path.join(folder_path, "*_u.csv")
    search_patternV = os.path.join(folder_path, "*_v.csv")
    
    i = 0
    for csv_fileU in glob.glob(search_patternU):
        csv_fileV = glob.glob(search_patternV)[i]
        i += 1
        
        # Read velocity components
        dfu = pd.read_csv(csv_fileU, index_col=0)
        dfv = pd.read_csv(csv_fileV, index_col=0)

        # Convert columns to float (X values) and index to float (Y values)
        x_values = dfu.columns.astype(float).values
        y_values = dfu.index.astype(float).values
        U = dfu.values
        V = dfv.values

        base_name = os.path.basename(csv_fileU)
        base_no_ext = os.path.splitext(base_name)[0]
        base_clean = base_no_ext.replace(".txt_u", "")

        # Define coarser grid size (adjust as needed)
        nx_coarse, ny_coarse = 20, 20  # Number of arrows in X and Y directions

        # Compute block-averaged velocity components
        U_avg, V_avg, factor_x, factor_y = block_average(U, V, nx_coarse, ny_coarse)

        # Generate coarser X, Y grid (Centering arrows over bins)
        x_bin_edges = np.linspace(x_values.min(), x_values.max(), nx_coarse + 1)
        y_bin_edges = np.linspace(y_values.min(), y_values.max(), ny_coarse + 1)

        x_coarse = (x_bin_edges[:-1] + x_bin_edges[1:]) / 2  # Midpoints in X
        y_coarse = (y_bin_edges[:-1] + y_bin_edges[1:]) / 2  # Midpoints in Y
        X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)

        # Plot the averaged velocity field with centered arrows
        plt.figure(figsize=(6, 6))
        plt.quiver(X_coarse, Y_coarse, U_avg, V_avg, color="b", scale=200)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Velocity Field Plot of \n{base_clean}")
        plt.grid()

        # Save the figure
        out_image_name = base_no_ext + "_velocity_field.png"
        out_path = os.path.join(save_folder, out_image_name)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()

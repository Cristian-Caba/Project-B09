#!/usr/bin/env python3

import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Import the dimensionalization function
from dimensionalization import convert_sx

# If you have a subplots helper, keep importing it:
from subplots_u_v import create_subplots

def main():
    folder_path = "PIV_planes"
    
    # We want to process all files ending in _u.csv, _v.csv, or _UV.csv
    file_patterns = ["*_u.csv", "*_v.csv", "*_UV.csv"]
    
    # Create a subfolder (images) to save our plots
    save_folder = os.path.join(folder_path, "images")
    os.makedirs(save_folder, exist_ok=True)

    for pattern in file_patterns:
        search_pattern = os.path.join(folder_path, pattern)
        for csv_file in glob.glob(search_pattern):
            # Read the CSV (first column as index -> Y-values)
            df = pd.read_csv(csv_file, index_col=0)

            # Convert columns (X) and index (Y) to float
            df.columns = df.columns.astype(float)
            df.index = df.index.astype(float)

            x_values = df.columns.values
            y_values = df.index.values

            # Boolean masks to select the data range:
            # X in [120, 155] and Y in [0, 4]
            x_mask = (x_values >= 120) & (x_values <= 155)
            y_mask = (y_values >= 0)   & (y_values <= 4)

            # Subset the DataFrame by position
            df_sub = df.iloc[y_mask, x_mask]

            # Extract the subset arrays
            x_sub = x_values[x_mask]
            y_sub = y_values[y_mask]
            Z_sub = df_sub.values

            base_name = os.path.basename(csv_file)

            # Identify velocity type and set color range
            if base_name.endswith("_u.csv"):
                color_label = "u Velocity"
                vmin, vmax = 0.0, 17.5
            elif base_name.endswith("_v.csv"):
                color_label = "v Velocity"
                vmin, vmax = -4.0, 1.0
            else:
                color_label = "Velocity Magnitude"
                vmin, vmax = None, None

            # 2) Convert your x_sub (mm) to dimensionless chord fraction using convert_sx
            #    ONLY do this if your x_sub makes sense as the “s_grid” input to convert_sx.
            x_dimless = convert_sx(x_sub)

            # Build meshgrid with dimensionless X and actual Y
            X_mesh, Y_mesh = np.meshgrid(x_dimless, y_sub)

            # Start a new figure
            plt.figure()
            contour = plt.contourf(X_mesh, Y_mesh, Z_sub, levels=50, vmin=vmin, vmax=vmax)
            plt.colorbar(contour, label=color_label)

            # 3) Set axis labels and title
            plt.xlabel("Dimensionless X (x / c)")
            plt.ylabel("Y (mm)")
            plt.title(f"Contour Plot of {color_label}\n{base_name}")

            # We still limit the Y plot from 0 to 4 mm
            plt.ylim([0, 4])

            # For the X-axis, we can show the dimensionless range of x_dimless.
            # If you want to fix the dimensionless range, do:
            plt.xlim([x_dimless.min(), x_dimless.max()])

            # Construct an output image name
            base_no_ext = os.path.splitext(base_name)[0]
            out_image_name = base_no_ext + "_dimX.png"
            out_path = os.path.join(save_folder, out_image_name)

            # Save and close figure
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved: {out_path}")

    # Optionally generate summary subplots if you have that logic
    create_subplots(folder_path, "u", "CC")
    create_subplots(folder_path, "v", "CC")
    create_subplots(folder_path, "u", "SC")
    create_subplots(folder_path, "v", "SC")


if __name__ == "__main__":
    main()
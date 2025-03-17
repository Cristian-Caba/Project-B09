#!/usr/bin/env python3

import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
            # Read the CSV (first column is index)
            df = pd.read_csv(csv_file, index_col=0)

            # Convert columns and index to float
            df.columns = df.columns.astype(float)
            df.index = df.index.astype(float)

            x_values = df.columns.values
            y_values = df.index.values

            # Boolean masks to select the data range:
            # X in [120, 155] and Y in [0, 4]
            x_mask = (x_values >= 120) & (x_values <= 155)
            y_mask = (y_values >= 0) & (y_values <= 4)

            # Subset the DataFrame by position using .iloc
            df_sub = df.iloc[y_mask, x_mask]

            # Get the actual subset arrays
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

            # Build meshgrid for plotting
            X_sub, Y_sub = np.meshgrid(x_sub, y_sub)

            # Start a new figure
            plt.figure()
            contour = plt.contourf(X_sub, Y_sub, Z_sub, levels=50, vmin=vmin, vmax=vmax)
            plt.colorbar(contour, label=color_label)

            # Set axis labels and title
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title(f"Contour Plot of {color_label}\n{base_name}")

            # Explicitly set the axis limits so they don't auto-scale
            plt.xlim([120, 155])
            plt.ylim([0, 4])

            # Construct an output image name
            base_no_ext = os.path.splitext(base_name)[0]
            out_image_name = base_no_ext + ".png"
            out_path = os.path.join(save_folder, out_image_name)

            # Save and close figure
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved: {out_path}")

    # Generate summary subplots
    create_subplots(folder_path, "u", "CC")
    create_subplots(folder_path, "v", "CC")
    create_subplots(folder_path, "u", "SC")
    create_subplots(folder_path, "v", "SC")


if __name__ == "__main__":
    main()

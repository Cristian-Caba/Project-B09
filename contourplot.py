#!/usr/bin/env python3

import glob
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    folder_path = "PIV_planes"
    
    # We want to process all files ending in _u.csv, _v.csv, or _UV.csv
    file_patterns = ["*_u.csv", "*_v.csv", "*_UV.csv"]
    
    # Create a subfolder (images) to save our plots
    save_folder = os.path.join(folder_path, "images")
    os.makedirs(save_folder, exist_ok=True)

    # Loop over each pattern and plot
    for pattern in file_patterns:
        # Build the full pattern (folder_path + filename pattern)
        search_pattern = os.path.join(folder_path, pattern)
        for csv_file in glob.glob(search_pattern):
            # Read the CSV, first column as the index
            df = pd.read_csv(csv_file, index_col=0)

            # Convert columns to float (X values) and index to float (Y values)
            x_values = df.columns.astype(float).values
            y_values = df.index.astype(float).values
            Z = df.values  # 2D array of velocity data (u, v, or magnitude)

            # Set up a label depending on which type of data we have
            base_name = os.path.basename(csv_file)  # e.g., "Case_CC_Span_1.txt_u.csv"
            if base_name.endswith("_u.csv"):
                color_label = "u Velocity"
            elif base_name.endswith("_v.csv"):
                color_label = "v Velocity"
            else:  # _UV.csv
                color_label = "Velocity Magnitude"

            # Create a meshgrid for contour plotting
            X, Y = np.meshgrid(x_values, y_values)

            # Start a new figure
            plt.figure()
            contour = plt.contourf(X, Y, Z)
            plt.colorbar(contour, label=color_label)

            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title(f"Contour Plot of {color_label}\n{base_name}")

            # Construct an output image name
            # Strip off the file extension, e.g. ".csv"
            base_no_ext = os.path.splitext(base_name)[0]  # "Case_CC_Span_1.txt_u"
            
            # (Optional) If you want to remove ".txt_u", ".txt_v", ".txt_UV" from the name, you can do:
            # base_clean = (base_no_ext
            #              .replace(".txt_u", "")
            #              .replace(".txt_v", "")
            #              .replace(".txt_UV", ""))
            #
            # But here, we'll just leave the suffix in the image name for clarity.

            out_image_name = base_no_ext + ".png"
            out_path = os.path.join(save_folder, out_image_name)

            # Save the figure as PNG
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()

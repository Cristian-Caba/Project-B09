#!/usr/bin/env python3

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_u_infinity(csv_file, y_target=3.5):
    """
    Reads the given *_u.csv file, converts row/column labels to float,
    finds the row closest to y_target, and returns the mean (across x)
    of the u velocity at that row.
    """
    df = pd.read_csv(csv_file, index_col=0)
    # Convert to float-based index/columns
    df.index = df.index.astype(float)   # Y-values
    df.columns = df.columns.astype(float)  # X-values

    # Find the row (Y) closest to y_target
    all_y = df.index.values
    idx_closest = np.argmin(np.abs(all_y - y_target))
    actual_y = all_y[idx_closest]

    # Extract that row and compute the mean
    row_data = df.loc[actual_y]
    u_inf = row_data.mean()

    return u_inf

def plot_dimensionless(csv_file, u_inf, out_folder="dimensionless_images"):
    """
    Reads a CSV file (either *_u.csv or *_v.csv), divides all velocity values
    by u_inf, restricts domain to x ∈ [120,155] and y ∈ [0,3.5],
    then creates a dimensionless contour plot (auto-scaled color limits).
    """
    df = pd.read_csv(csv_file, index_col=0)
    # Convert index/columns to float
    df.index = df.index.astype(float)
    df.columns = df.columns.astype(float)

    # Create arrays of X and Y from columns/index
    x_vals = df.columns.values
    y_vals = df.index.values

    # Domain mask
    x_mask = (x_vals >= 120) & (x_vals <= 155)
    y_mask = (y_vals >= 0) & (y_vals <= 3.5)

    # Subset by position using .iloc
    df_sub = df.iloc[y_mask, x_mask]
    x_sub = x_vals[x_mask]
    y_sub = y_vals[y_mask]

    # Convert to dimensionless: velocity / u_inf
    Z_sub = df_sub.values / u_inf

    # Identify which velocity component we have based on the file name
    base_name = os.path.basename(csv_file)
    if base_name.endswith("_u.csv"):
        velocity_label = "u/U∞"
    else:
        velocity_label = "v/U∞"

    # Create the meshgrid
    X_sub, Y_sub = np.meshgrid(x_sub, y_sub)

    # Make sure output folder exists
    os.makedirs(out_folder, exist_ok=True)

    # Plot
    plt.figure()
    cont = plt.contourf(X_sub, Y_sub, Z_sub, levels=50)
    plt.colorbar(cont, label=velocity_label)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Dimensionless {velocity_label}\n{base_name}")

    # Fix axes so they only show x ∈ [120,155] and y ∈ [0,3.5]
    plt.xlim([120, 155])
    plt.ylim([0, 3.5])

    # Save figure
    base_no_ext = os.path.splitext(base_name)[0]
    out_png = os.path.join(out_folder, base_no_ext + "_dimensionless.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_png}")

def main():
    """
    Loops through each config 'CC' or 'SC' and span 1..24.
    1) Finds the file for the u-velocity (Case_<config>_Span_<i>.txt_u.csv).
    2) Computes the free-stream velocity u_inf at y=3.5.
    3) Plots the dimensionless u and v in the domain [120..155] x [0..3.5].
    """
    # Adjust these if your folder or filenames differ
    folder = "PIV_planes"
    configs = ["CC", "SC"]
    span_numbers = range(1, 25)

    for config in configs:
        for span in span_numbers:
            # Build the file paths for the u and v CSV files
            u_file = os.path.join(folder, f"Case_{config}_Span_{span}.txt_u.csv")
            v_file = os.path.join(folder, f"Case_{config}_Span_{span}.txt_v.csv")

            if not (os.path.isfile(u_file) and os.path.isfile(v_file)):
                # If either file doesn't exist, skip
                print(f"Skipping span {span}, config {config} because file not found")
                continue

            # 1) Compute the free-stream velocity from the u-file
            u_inf = compute_u_infinity(u_file, y_target=3.5)

            # 2) Plot dimensionless u and v
            plot_dimensionless(u_file, u_inf, out_folder="dimensionless_images")
            plot_dimensionless(v_file, u_inf, out_folder="dimensionless_images")

if __name__ == "__main__":
    main()
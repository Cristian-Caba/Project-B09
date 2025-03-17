import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re  # Make sure to import re

def extract_numeric(filename):
    """Extracts the numeric span value from a filename like 'Case_CC_Span_12.txt_u.csv'."""
    match = re.search(r"Span_(\d+)", filename)
    return int(match.group(1)) if match else float('inf')  # Default to 'inf' if no match

def create_subplots(folder_path, velocity_type, case):
    """
    Creates a single figure with 24 subplots for a given velocity type (u or v) and case (CC or SC),
    using a dynamically scaled shared color bar.
    """
    # Define search pattern to find relevant CSV files
    search_pattern = os.path.join(folder_path, f"*_{velocity_type}.csv")

    # Filter files correctly based on case
    csv_files = [f for f in glob.glob(search_pattern) if case in f]  # Ensure only CC or SC files
    csv_files = sorted(csv_files, key=extract_numeric)  # Sort numerically by span
    csv_files = csv_files[:24]  # Keep only the first 24 files

    if len(csv_files) != 24:
        print(f"Warning: Expected 24 files but found {len(csv_files)} for {velocity_type} {case}")

    # Step 1: Determine global min/max across all datasets
    global_min, global_max = float("inf"), float("-inf")

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, index_col=0)
        df.columns = df.columns.astype(float)
        df.index = df.index.astype(float)

        # Apply the masks
        x_values = df.columns.values
        y_values = df.index.values
        x_mask = (x_values >= 120) & (x_values <= 155)
        y_mask = (y_values >= 0) & (y_values <= 4)
        df_sub = df.iloc[y_mask, x_mask]

        # Update global min/max
        local_min, local_max = df_sub.min().min(), df_sub.max().max()
        global_min = min(global_min, local_min)
        global_max = max(global_max, local_max)

    print(f"{velocity_type.upper()} {case} - Global Min: {global_min}, Global Max: {global_max}")

    # Step 2: Create subplots with dynamically adjusted color scale
    fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(18, 12), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten to index subplots easily

    contour_plots = []  # Store contour references

    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file, index_col=0)
        df.columns = df.columns.astype(float)
        df.index = df.index.astype(float)

        # Apply the masks
        x_values = df.columns.values
        y_values = df.index.values
        x_mask = (x_values >= 120) & (x_values <= 155)
        y_mask = (y_values >= 0) & (y_values <= 4)
        df_sub = df.iloc[y_mask, x_mask]
        x_sub = x_values[x_mask]
        y_sub = y_values[y_mask]
        Z_sub = df_sub.values

        X_sub, Y_sub = np.meshgrid(x_sub, y_sub)

        # Plot contour in subplot
        ax = axes[i]
        contour = ax.contourf(X_sub, Y_sub, Z_sub, levels=50, vmin=global_min, vmax=global_max, cmap="viridis")
        contour_plots.append(contour)

    # Add title to each subplot
        ax.set_title(os.path.basename(csv_file), fontsize=8)

        # Ensure ticks are visible
        ax.set_xticks(np.linspace(min(x_sub), max(x_sub), 5))  # Adjust number of ticks
        ax.set_yticks(np.linspace(min(y_sub), max(y_sub), 5))

        ax.tick_params(axis='both', which='major', labelsize=8)  # Small tick labels for clarity

    # Add shared X and Y labels
    fig.text(0.5, 0.05, "X Coordinate", ha="center", fontsize=12)
    fig.text(0.04, 0.5, "Y Coordinate", va="center", rotation="vertical", fontsize=12)

    # Adjust subplot spacing to prevent overlap
    fig.subplots_adjust(left=0.1, right=0.88, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)

    # Add color bar
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])  
    fig.colorbar(contour_plots[0], cax=cbar_ax, label=f"{velocity_type.upper()} Velocity")

    fig.suptitle(f"Contour Plots for {velocity_type.upper()} ({case})", fontsize=16)

    # Save the figure
    save_path = os.path.join(folder_path, f"summary_{velocity_type}_{case}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved summary plot: {save_path}")

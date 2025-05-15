#!/usr/bin/env python3

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

from dimensionalization import convert_sx  # interpret x-values as arc-length s => dimensionless (x/c_x)

###############################################################################
def extract_freestream_u(folder, config, span_index, y_fs=3.5):
    """
    Loads 'Case_{config}_Span_{span_index}.txt_u.csv',
    finds the row nearest y_fs, returns a Pandas Series:
      index = X, values = freestream velocity (u) at y=y_fs.
    Returns None if missing or empty.
    """
    fname = f"Case_{config}_Span_{span_index}.txt_u.csv"
    path = os.path.join(folder, fname)
    if not os.path.isfile(path):
        print(f"[extract_freestream_u] Missing file: {path}")
        return None

    df = pd.read_csv(path, index_col=0)
    if df.empty:
        print(f"[extract_freestream_u] Empty data in {path}")
        return None

    df.index = df.index.astype(float)   # Y
    df.columns = df.columns.astype(float)  # X
    all_y = df.index.values
    if len(all_y) == 0:
        print(f"[extract_freestream_u] No Y data in {path}")
        return None

    idx_closest = np.argmin(np.abs(all_y - y_fs))
    actual_y = all_y[idx_closest]

    row_fs = df.loc[actual_y]  # Series: index=X, values=u
    row_fs = row_fs.sort_index()
    return row_fs

def create_compound_plots(all_data, config, save_folder, min_max_uv):
    """
    Create compound plots for a given configuration (CC or SC)
    all_data: Dictionary containing processed data for all span indices
    """
    # Exclude span 25 and sort remaining spans
    span_indices = sorted([idx for idx in all_data.keys() if idx != 25])
    if not span_indices:  # Skip if no data for this configuration
        print(f"No data available for {config} configuration (excluding span 25)")
        return
        
    num_spans = len(span_indices)
    
    for component in ['u', 'v']:
        # Check if we have data for this component
        if component not in all_data[span_indices[0]]:
            print(f"No {component} data available for {config} configuration")
            continue
            
        # Create figure with 6x4 grid
        fig = plt.figure(figsize=(20, 24))
        gs = GridSpec(6, 4, figure=fig)
        
        for i, span_idx in enumerate(span_indices):
            row = i // 4
            col = i % 4
            ax = fig.add_subplot(gs[row, col])
            
            data = all_data[span_idx][component]
            contour = ax.contourf(
                data['X_mesh'],
                data['Y_mesh'],
                data['dimZ_sub'],
                levels=50,
                cmap='Spectral',
                vmin=float(min_max_uv[0 if component == 'u' else 1][0]),
                vmax=float(min_max_uv[0 if component == 'u' else 1][1])
            )
            
            if config == "SC":
                width_dim = 1.4/900
                spacing_dim = 9.2/900
                height_mm = 0.17
                rectLeft = [0.125]
                center0 = rectLeft[0] + width_dim/2
                for k in range(1, 5):
                    center_k = center0 + k*spacing_dim
                    left_k = center_k - width_dim/2
                    rectLeft.append(left_k)
                
                for left in rectLeft:
                    rect = patches.Rectangle(
                        (left, 0.0),
                        width_dim,
                        height_mm,
                        color='black'
                    )
                    ax.add_patch(rect)
            
            ax.set_title(f"Span {span_idx}")
            ax.set_xlabel("x/c_x" if row == 5 else "")
            ax.set_ylabel("Y [mm]" if col == 0 else "")
            ax.set_xlim([data['x_dim_min'], data['x_dim_max']])
            ax.set_ylim([data['y_min'], data['y_max']])
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(contour, cax=cbar_ax, label=f"{component}/u∞(x)")
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        out_path = os.path.join(save_folder, f"Compound_{config}_{component}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved compound plot: {out_path}")
###############################################################################
def main():
    folder_path = "PIV_planes"

    # Patterns for _u.csv, _v.csv, _UV.csv
    file_patterns = ["*_u.csv", "*_v.csv"]
    compound_data = {'CC': {}, 'SC': {}}

    # Subfolder for dimensionless images
    save_folder = os.path.join(folder_path, "images_dimless_x")
    os.makedirs(save_folder, exist_ok=True)

    # Restrict domain
    x_min, x_max = 120, 155
    y_min, y_max = 0, 3

    min_max_uv = [[0,1], [-0.12, 0]]

    from matplotlib.colors import LinearSegmentedColormap

    red_white_cmap = LinearSegmentedColormap.from_list("red_white", ["white", "red"])
    blue_white_cmap = LinearSegmentedColormap.from_list("blue_white", ["blue", "white"])

    cmaps = [red_white_cmap, blue_white_cmap]

    i = 0

    for pattern in file_patterns:
        search_pattern = os.path.join(folder_path, pattern)
        for csv_file in glob.glob(search_pattern):
            base_name = os.path.basename(csv_file)

            # Read
            df = pd.read_csv(csv_file, index_col=0)
            if df.empty:
                print(f"[main] Empty data in {csv_file}, skipping.")
                continue

            df.index = df.index.astype(float)   # Y
            df.columns = df.columns.astype(float)  # X

            # Restrict
            x_vals = df.columns.values
            y_vals = df.index.values

            x_mask = (x_vals >= x_min) & (x_vals <= x_max)
            y_mask = (y_vals >= y_min) & (y_vals <= y_max)

            df_sub = df.iloc[y_mask, x_mask]
            if df_sub.empty:
                print(f"[main] After restricting domain, no data in {csv_file}, skipping.")
                continue

            x_sub = df_sub.columns.values
            y_sub = df_sub.index.values
            Z_sub = df_sub.values

            # Parse config + span
            try:
                tokens = base_name.split("_")
                config = tokens[1]  # "CC" or "SC"
                span_str = tokens[3].replace(".txt", "")
                span_index = int(span_str)
            except:
                print(f"[main] Cannot parse config/span from {base_name}, skipping dimensionless step.")
                continue

            # Extract free-stream from the *u.csv* file
            fs_row = extract_freestream_u(folder_path, config, span_index, y_fs=3.5)
            if fs_row is None:
                print(f"[main] No free-stream data for {base_name}, skipping dimensionless.")
                continue

            # Build dimensionless velocity array
            fs_xvals = fs_row.index.values
            dimZ_sub = np.full_like(Z_sub, np.nan, dtype=float)

            for ix, xval in enumerate(x_sub):
                idx_closest = np.argmin(np.abs(fs_xvals - xval))
                fs_val = fs_row[fs_xvals[idx_closest]]
                if fs_val == 0 or np.isnan(fs_val):
                    continue
                dimZ_sub[:, ix] = Z_sub[:, ix] / fs_val

            # Dimensionless x-axis
            x_dim_sub = convert_sx(x_sub)
            X_mesh, Y_mesh = np.meshgrid(x_dim_sub, y_sub)

            # Identify velocity type
            if base_name.endswith("_u.csv"):
                raw_label = "u"
            elif base_name.endswith("_v.csv"):
                raw_label = "v"
            else:
                raw_label = "Velocity Mag"
            if span_index not in compound_data[config]:
                compound_data[config][span_index] = {}
            
            x_dim_min = np.min(convert_sx(np.array([x_min])))
            x_dim_max = np.min(convert_sx(np.array([x_max])))
            
            compound_data[config][span_index][raw_label] = {
                'X_mesh': X_mesh,
                'Y_mesh': Y_mesh,
                'dimZ_sub': dimZ_sub,
                'x_dim_min': x_dim_min,
                'x_dim_max': x_dim_max,
                'y_min': y_min,
                'y_max': y_max
            }

            # MARKED CHANGE: Simply fix the color scale + 'bwr' colormap
            # Example: from -2.5 to +3.0
            plt.figure(figsize=(9,3))
            plt.tight_layout()
            contour = plt.contourf(
                X_mesh,
                Y_mesh,
                dimZ_sub,
                levels=50,
                cmap='Spectral', #cmaps[i],     Blue-White-Red scheme
                vmin=float(min_max_uv[i][0]),     # Lower limit
                vmax=float(min_max_uv[i][1])      # Upper limit
            )
            # 1) Import patches if not already done at the top of your file:
            import matplotlib.patches as patches
            ax = plt.gca()

            # ADD RECTANGLES FOR SC ONLY
            if config == "SC":
                # Convert lengths from mm to dimensionless for the X direction
                width_dim = 1.4/900
                spacing_dim = 9.2/900
                height_mm = 0.17  # Y in mm

                # Left edge of the first rectangle is x/c=0.125
                rectLeft = [0.125]  # dimensionless

                center0 = rectLeft[0] + width_dim/2
                # Build left edges for rectangles #2..#5
                for k in range(1, 5):
                    center_k = center0 + k*spacing_dim
                    left_k = center_k - width_dim/2
                    rectLeft.append(left_k)

                # Add each rectangle (all black)
                for left in rectLeft:
                    rect = patches.Rectangle(
                        (left, 0.0),  # bottom-left corner
                        width_dim,    # dimensionless width
                        height_mm,    # height in mm
                        color='black'
                    )
                    ax.add_patch(rect)

            plt.colorbar(contour, label=f"{raw_label}/u∞(x)")

            plt.xlabel("$x/c_x$")
            plt.ylabel("y [mm]")
            #plt.title(f"Dimensionless {raw_label}\nPlane={span_index}, Config={config}")

            # Fix axis range in dimensionless X, Y in [0..4]
            x_dim_min = np.min(convert_sx(np.array([x_min])))
            x_dim_max = np.min(convert_sx(np.array([x_max])))
            plt.xlim([x_dim_min, x_dim_max])
            plt.ylim([y_min, y_max])

            out_name = os.path.splitext(base_name)[0] + "_dimless_x.png"
            out_path = os.path.join(save_folder, out_name)
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved dimensionless velocity + dimensionless X plot: {out_path}")
        i += 1
    for config in ['CC', 'SC']:
        if compound_data[config]:  # Only if we have data for this configuration
            create_compound_plots(compound_data[config], config, save_folder, min_max_uv)


if __name__ == "__main__":
    main()

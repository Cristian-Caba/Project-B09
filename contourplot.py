#!/usr/bin/env python3

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

###############################################################################
def main():
    folder_path = "PIV_planes"

    # Patterns for _u.csv, _v.csv, _UV.csv
    file_patterns = ["*_u.csv", "*_v.csv", "*_UV.csv"]

    # Subfolder for dimensionless images
    save_folder = os.path.join(folder_path, "images_dimless_x")
    os.makedirs(save_folder, exist_ok=True)

    # Restrict domain
    x_min, x_max = 120, 155
    y_min, y_max = 0, 4

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

            # MARKED CHANGE: Simply fix the color scale + 'bwr' colormap
            # Example: from -2.5 to +3.0
            vmin = 0
            vmax = 1

            plt.figure()
            contour = plt.contourf(
                X_mesh,
                Y_mesh,
                dimZ_sub,
                levels=50,
                cmap='bwr',    # Blue-White-Red scheme
                vmin=vmin,     # Lower limit
                vmax=vmax      # Upper limit
            )
            plt.colorbar(contour, label=f"{raw_label}/u∞(x)")

            plt.xlabel("Dimensionless X (x/c_x)")
            plt.ylabel("Y [mm]")
            plt.title(f"Dimensionless {raw_label}\nPlane={span_index}, Config={config}")

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

if __name__ == "__main__":
    main()

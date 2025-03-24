#!/usr/bin/env python3

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dimensionalization import convert_sx  # interpret x-values as arc-length s => dimensionless (x/c_x)


def extract_freestream_u(folder, config, span_index, y_fs=3.5):
    """
    Loads the 'Case_{config}_Span_{span_index}.txt_u.csv' file,
    finds the row nearest y_fs, and returns it as a Pandas Series
    with index = X, value = freestream velocity (u) at y=y_fs.
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

    # Convert index & columns to float
    df.index = df.index.astype(float)    # Y
    df.columns = df.columns.astype(float)  # X

    all_y = df.index.values
    if len(all_y) == 0:
        print(f"[extract_freestream_u] No Y data in {path}")
        return None

    # Find row nearest y_fs
    idx_closest = np.argmin(np.abs(all_y - y_fs))
    actual_y = all_y[idx_closest]

    row_fs = df.loc[actual_y]  # Series: index = X, value = u-velocity
    row_fs = row_fs.sort_index()
    return row_fs

def main():
    folder_path = "PIV_planes"

    # Patterns for _u.csv, _v.csv, maybe _UV.csv
    file_patterns = ["*_u.csv", "*_v.csv", "*_UV.csv"]

    # Subfolder for dimensionless images
    save_folder = os.path.join(folder_path, "images_dimless_x")
    os.makedirs(save_folder, exist_ok=True)

    # We restrict the domain to x in [120..155], y in [0..4]
    x_min, x_max = 120, 155
    y_min, y_max = 0, 4

    for pattern in file_patterns:
        search_pattern = os.path.join(folder_path, pattern)
        for csv_file in glob.glob(search_pattern):
            base_name = os.path.basename(csv_file)  # e.g. "Case_CC_Span_1.txt_u.csv"

            # Read the data
            df = pd.read_csv(csv_file, index_col=0)
            if df.empty:
                print(f"[main] Empty data in {csv_file}, skipping.")
                continue

            # Convert to float
            df.index = df.index.astype(float)    # Y
            df.columns = df.columns.astype(float)  # X

            # Subset to x in [120..155], y in [0..4]
            x_vals = df.columns.values
            y_vals = df.index.values

            x_mask = (x_vals >= x_min) & (x_vals <= x_max)
            y_mask = (y_vals >= y_min) & (y_vals <= y_max)

            df_sub = df.iloc[y_mask, x_mask]
            if df_sub.empty:
                print(f"[main] After restricting domain, no data in {csv_file}, skipping.")
                continue

            # Prepare arrays
            x_sub = df_sub.columns.values  # restricted X
            y_sub = df_sub.index.values    # restricted Y
            Z_sub = df_sub.values          # 2D velocity array

            # Attempt to parse config + span index from filename
            # e.g. "Case_CC_Span_1.txt_u.csv" => tokens = ["Case", "CC", "Span", "1.txt", "u.csv"]
            try:
                tokens = base_name.split("_")
                config = tokens[1]   # "CC" or "SC"
                span_str = tokens[3].replace(".txt", "")  # "1"
                span_index = int(span_str)
            except:
                print(f"[main] Cannot parse config/span from {base_name}, skipping dimensionless conversion.")
                # Just do a raw plot if you like. We'll skip dimensionless step here:
                continue

            # 1) Extract the free-stream row from the *u.csv* for this plane
            fs_row = extract_freestream_u(folder_path, config, span_index, y_fs=3.5)
            if fs_row is None:
                print(f"[main] No free-stream data for {base_name}, skipping dimensionless.")
                continue

            # 2) Build dimensionless velocity array
            #    For each column x_sub[i], find nearest X in fs_row.index, then do Z_sub[:,i] / fs_val
            fs_xvals = fs_row.index.values
            dimZ_sub = np.full_like(Z_sub, np.nan, dtype=float)  # dimensionless velocity

            for ix, xval in enumerate(x_sub):
                # find nearest X in fs_xvals
                idx_closest = np.argmin(np.abs(fs_xvals - xval))
                fs_x = fs_xvals[idx_closest]
                fs_val = fs_row[fs_x]
                if fs_val == 0 or np.isnan(fs_val):
                    # avoid dividing by zero
                    continue
                # dimensionless velocity
                dimZ_sub[:, ix] = Z_sub[:, ix] / fs_val

            # 3) Dimensionless x-axis
            #    interpret x_sub as arc-length s, and do convert_sx(s_grid)
            x_dim_sub = convert_sx(x_sub)  # dimensionless x

            # We'll meshgrid dimensionless X with the actual Y
            X_mesh, Y_mesh = np.meshgrid(x_dim_sub, y_sub)

            # Identify velocity type
            if base_name.endswith("_u.csv"):
                raw_label = "u"
            elif base_name.endswith("_v.csv"):
                raw_label = "v"
            else:
                raw_label = "Velocity Magnitude"

            # 4) Plot dimensionless velocity vs. dimensionless X, restricted Y
            plt.figure()
            contour = plt.contourf(X_mesh, Y_mesh, dimZ_sub, levels=50)
            plt.colorbar(contour, label=f"{raw_label}/u∞(x)")

            plt.xlabel("Dimensionless X (x/c_x)")
            plt.ylabel("Y [mm]")
            plt.title(f"Dimensionless {raw_label}\nPlane={span_index}, Config={config}")

            # We'll fix Y in [0..4] dimensionless X in [ convert_sx(120) .. convert_sx(155) ]
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

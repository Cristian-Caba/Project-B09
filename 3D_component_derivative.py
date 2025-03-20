#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_plane(u_file, v_file, x_min=120, x_max=155, y_min=0, y_max=4):
    """
    Reads the u and v files for a single plane, returns:
        x_values, y_values, u_data, v_data
    restricted to the domain x ∈ [x_min, x_max] and y ∈ [y_min, y_max].
    If a file is missing or there's no valid domain, returns None.
    """
    if not os.path.isfile(u_file) or not os.path.isfile(v_file):
        print(f"Missing file(s): {u_file} or {v_file}")
        return None

    df_u = pd.read_csv(u_file, index_col=0)
    df_v = pd.read_csv(v_file, index_col=0)

    # Convert index (Y) and columns (X) to float
    df_u.index = df_u.index.astype(float)
    df_u.columns = df_u.columns.astype(float)
    df_v.index = df_v.index.astype(float)
    df_v.columns = df_v.columns.astype(float)

    # 1) Intersect domain
    common_x = np.intersect1d(df_u.columns, df_v.columns)
    common_y = np.intersect1d(df_u.index, df_v.index)
    if len(common_x) == 0 or len(common_y) == 0:
        print(f"No overlapping domain for {u_file} and {v_file}")
        return None

    # 2) Restrict to [x_min..x_max], [y_min..y_max]
    x_mask = (common_x >= x_min) & (common_x <= x_max)
    y_mask = (common_y >= y_min) & (common_y <= y_max)
    restricted_x = common_x[x_mask]
    restricted_y = common_y[y_mask]

    if len(restricted_x) < 3 or len(restricted_y) < 3:
        # Need at least 3 points in each direction to do central difference
        print(f"Domain too small after restriction for: {u_file} / {v_file}")
        return None

    # 3) Subset
    df_u_sub = df_u.loc[restricted_y, restricted_x].sort_index(axis=0).sort_index(axis=1)
    df_v_sub = df_v.loc[restricted_y, restricted_x].sort_index(axis=0).sort_index(axis=1)

    x_vals = df_u_sub.columns.values
    y_vals = df_u_sub.index.values
    u_data = df_u_sub.values
    v_data = df_v_sub.values

    return x_vals, y_vals, u_data, v_data

def compute_dw_dz(u_data, v_data, dx, dy):
    """
    Given 2D arrays u_data, v_data of shape (ny, nx),
    compute dw/dz using continuity:
        dw/dz = -[ d(u)/dx + d(v)/dy ]
    via central difference. Returns a (ny-2, nx-2) array.
    """
    ny, nx = u_data.shape
    # du/dx → shape (ny, nx-2)
    du_dx = (u_data[:, 2:] - u_data[:, :-2]) / (2.0 * dx)
    # dv/dy → shape (ny-2, nx)
    dv_dy = (v_data[2:, :] - v_data[:-2, :]) / (2.0 * dy)

    # Overlapping interior region is (ny-2, nx-2)
    du_dx_inner = du_dx[1:-1, :]
    dv_dy_inner = dv_dy[:, 1:-1]

    dw_dz = - (du_dx_inner + dv_dy_inner)
    return dw_dz

def plot_dw_dz(x_values, y_values, dw_dz, plane_label,
               out_folder="dw_dz_images",
               dw_min=-2.5, dw_max=3.0):
    """
    Creates a contour plot of dw/dz on the interior domain with
    a fixed color scale from dw_min to dw_max.
    """
    os.makedirs(out_folder, exist_ok=True)

    # Because dw_dz is smaller by one row/column on each boundary:
    x_inner = x_values[1:-1]
    y_inner = y_values[1:-1]

    X, Y = np.meshgrid(x_inner, y_inner)

    # We define 50 levels between dw_min and dw_max for a consistent scale.
    levels = np.linspace(dw_min, dw_max, 51)

    plt.figure()
    cont = plt.contourf(X, Y, dw_dz, levels=levels, vmin=dw_min, vmax=dw_max)
    plt.colorbar(cont, label="dw/dz")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"dw/dz for {plane_label}")

    # Fix displayed x, y range to exactly show [x_inner.min..x_inner.max], etc.
    plt.xlim([x_inner.min(), x_inner.max()])
    plt.ylim([y_inner.min(), y_inner.max()])

    out_name = f"dw_dz_{plane_label}.png"
    out_path = os.path.join(out_folder, out_name)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

def main():
    """
    Loop over planes 1..24 for CC and SC configurations.
    Restrict domain to x in [120..155], y in [0..4].
    Compute dw/dz from continuity. Plot with a fixed color scale
    from -2.5 to 3.0 in dw/dz.
    """
    folder = "PIV_planes"
    out_folder = "dw_dz_images"
    configs = ["CC", "SC"]
    dx, dy = 1.0, 1.0  # mm spacing

    for config in configs:
        for i in range(1, 25):
            plane_label = f"{config}_Span_{i}"
            u_file = os.path.join(folder, f"Case_{config}_Span_{i}.txt_u.csv")
            v_file = os.path.join(folder, f"Case_{config}_Span_{i}.txt_v.csv")

            plane_data = read_plane(u_file, v_file,
                                    x_min=120, x_max=155,
                                    y_min=0,   y_max=4)
            if plane_data is None:
                print(f"Skipping {plane_label} (no valid data).")
                continue

            x_vals, y_vals, u_data, v_data = plane_data

            # Compute dw/dz
            dw_dz_array = compute_dw_dz(u_data, v_data, dx, dy)

            # Plot with fixed color scale from -2.5 to 3
            plot_dw_dz(x_vals, y_vals, dw_dz_array, plane_label,
                       out_folder=out_folder,
                       dw_min=-2.5, dw_max=3.0)

if __name__ == "__main__":
    main()
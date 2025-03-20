#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dimensionalization import convert_sx  # for dimensionless X-axis

def difference_and_plot(cc_u_file, sc_u_file, cc_v_file, sc_v_file, 
                        output_folder="images_diff", 
                        x_min=120, x_max=155, 
                        y_min=0,   y_max=4):
    """
    Reads clean config (CC) and strips config (SC) data for U and V,
    subtracts (SC - CC) for both U and V, 
    restricts domain to x ∈ [x_min, x_max], y ∈ [y_min, y_max],
    converts X to dimensionless chord fraction, and makes contour plots.
    """
    # Ensure output folder
    os.makedirs(output_folder, exist_ok=True)

    # --- 1) Load CC and SC for U ---
    df_cc_u = pd.read_csv(cc_u_file, index_col=0)
    df_sc_u = pd.read_csv(sc_u_file, index_col=0)
    
    # Convert columns (X) and index (Y) to float
    df_cc_u.columns = df_cc_u.columns.astype(float)
    df_cc_u.index   = df_cc_u.index.astype(float)
    df_sc_u.columns = df_sc_u.columns.astype(float)
    df_sc_u.index   = df_sc_u.index.astype(float)

    # Restrict domain [x_min..x_max], [y_min..y_max] by position
    x_vals_u = df_cc_u.columns.values
    y_vals_u = df_cc_u.index.values

    x_mask_u = (x_vals_u >= x_min) & (x_vals_u <= x_max)
    y_mask_u = (y_vals_u >= y_min) & (y_vals_u <= y_max)

    df_cc_u_sub = df_cc_u.iloc[y_mask_u, x_mask_u]
    df_sc_u_sub = df_sc_u.iloc[y_mask_u, x_mask_u]

    # Compute difference SC - CC for U
    diff_u = df_sc_u_sub.values - df_cc_u_sub.values

    # Build dimensionless X
    x_sub_u = df_cc_u_sub.columns.values  # restricted x
    y_sub_u = df_cc_u_sub.index.values    # restricted y
    x_dimless_u = convert_sx(x_sub_u)

    Xdim_u, Ydim_u = np.meshgrid(x_dimless_u, y_sub_u)

    # Plot difference in U
    plt.figure()
    cont_u = plt.contourf(Xdim_u, Ydim_u, diff_u, levels=50)
    plt.colorbar(cont_u, label="ΔU (SC - CC)")
    plt.xlabel("Dimensionless X (x/c)")
    plt.ylabel("Y")
    plt.title(f"Difference in U Velocity\n(SC - CC): {os.path.basename(cc_u_file)}")

    out_u = os.path.join(output_folder, f"difference_U_{os.path.basename(cc_u_file)}_dimx.png")
    plt.savefig(out_u, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_u}")

    # --- 2) Load CC and SC for V ---
    df_cc_v = pd.read_csv(cc_v_file, index_col=0)
    df_sc_v = pd.read_csv(sc_v_file, index_col=0)

    df_cc_v.columns = df_cc_v.columns.astype(float)
    df_cc_v.index   = df_cc_v.index.astype(float)
    df_sc_v.columns = df_sc_v.columns.astype(float)
    df_sc_v.index   = df_sc_v.index.astype(float)

    # Restrict domain
    x_vals_v = df_cc_v.columns.values
    y_vals_v = df_cc_v.index.values

    x_mask_v = (x_vals_v >= x_min) & (x_vals_v <= x_max)
    y_mask_v = (y_vals_v >= y_min) & (y_vals_v <= y_max)

    df_cc_v_sub = df_cc_v.iloc[y_mask_v, x_mask_v]
    df_sc_v_sub = df_sc_v.iloc[y_mask_v, x_mask_v]

    # Compute difference SC - CC for V
    diff_v = df_sc_v_sub.values - df_cc_v_sub.values

    # Build dimensionless X
    x_sub_v = df_cc_v_sub.columns.values
    y_sub_v = df_cc_v_sub.index.values
    x_dimless_v = convert_sx(x_sub_v)

    Xdim_v, Ydim_v = np.meshgrid(x_dimless_v, y_sub_v)

    # Plot difference in V
    plt.figure()
    cont_v = plt.contourf(Xdim_v, Ydim_v, diff_v, levels=50)
    plt.colorbar(cont_v, label="ΔV (SC - CC)")
    plt.xlabel("Dimensionless X (x/c)")
    plt.ylabel("Y")
    plt.title(f"Difference in V Velocity\n(SC - CC): {os.path.basename(cc_v_file)}")

    out_v = os.path.join(output_folder, f"difference_V_{os.path.basename(cc_v_file)}_dimx.png")
    plt.savefig(out_v, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_v}")

def main():
    """
    Generate difference plots (SC - CC) for Span_1..24,
    restricting to x∈[120,155], y∈[0,4], 
    and converting X to dimensionless chord fraction.
    """
    for i in range(1, 25):
        cc_u_file = f"PIV_planes/Case_CC_Span_{i}.txt_u.csv"
        sc_u_file = f"PIV_planes/Case_SC_Span_{i}.txt_u.csv"
        cc_v_file = f"PIV_planes/Case_CC_Span_{i}.txt_v.csv"
        sc_v_file = f"PIV_planes/Case_SC_Span_{i}.txt_v.csv"

        difference_and_plot(
            cc_u_file=cc_u_file,
            sc_u_file=sc_u_file,
            cc_v_file=cc_v_file,
            sc_v_file=sc_v_file,
            output_folder="images_diff",
            x_min=120,
            x_max=155,
            y_min=0,
            y_max=4
        )

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contourplot  # Import your existing contourplot.py module

def difference_and_plot(cc_u_file, sc_u_file, cc_v_file, sc_v_file, output_folder="images_diff"):
    """
    Reads clean configuration (CC) and strips configuration (SC) data for both U and V.
    Subtracts CC from SC for each velocity component (U, V) and creates contour plots
    of the resulting differences.
    """
    # Make sure our output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # --- 1) Load CC and SC for U ---
    df_cc_u = pd.read_csv(cc_u_file, index_col=0)
    df_sc_u = pd.read_csv(sc_u_file, index_col=0)

    # Convert columns/index to float for contour plotting
    x_values_u = df_cc_u.columns.astype(float).values
    y_values_u = df_cc_u.index.astype(float).values

    # --- 2) Compute U difference: SC - CC ---
    diff_u = df_sc_u.values - df_cc_u.values

    # --- 3) Plot the difference in U ---
    X_u, Y_u = np.meshgrid(x_values_u, y_values_u)
    plt.figure()
    cont_u = plt.contourf(X_u, Y_u, diff_u)
    plt.colorbar(cont_u, label="ΔU (SC - CC)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Difference in U Velocity (SC - CC)\n{os.path.basename(cc_u_file)}")
    out_u = os.path.join(output_folder, f"difference_U_{os.path.basename(cc_u_file)}.png")
    plt.savefig(out_u, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_u}")

    # --- 4) Load CC and SC for V ---
    df_cc_v = pd.read_csv(cc_v_file, index_col=0)
    df_sc_v = pd.read_csv(sc_v_file, index_col=0)

    x_values_v = df_cc_v.columns.astype(float).values
    y_values_v = df_cc_v.index.astype(float).values

    # --- 5) Compute V difference: SC - CC ---
    diff_v = df_sc_v.values - df_cc_v.values

    # --- 6) Plot the difference in V ---
    X_v, Y_v = np.meshgrid(x_values_v, y_values_v)
    plt.figure()
    cont_v = plt.contourf(X_v, Y_v, diff_v)
    plt.colorbar(cont_v, label="ΔV (SC - CC)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Difference in V Velocity (SC - CC)\n{os.path.basename(cc_v_file)}")
    out_v = os.path.join(output_folder, f"difference_V_{os.path.basename(cc_v_file)}.png")
    plt.savefig(out_v, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_v}")

def main():
    """
    Example usage: Generate difference plots for Span_1 through Span_24.
    Adjust if you have a different numbering range or file naming convention.
    """
    # Loop over the integer index from 1 to 24
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
            output_folder="images_diff"
        )

if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dimensionalization import convert_sx  # interpret x-values as arc-length s for dimensionless

def extract_u_at_y(csv_file, y_target=3.5):
    """
    Reads a *_u.csv file (columns = X, index = Y).
    Finds the row nearest y_target, returns it as a Pandas Series:
      series.index = X array (floats),
      series.values = U velocities at that Y.
    Returns None if file is missing/empty or row can't be found.
    """
    if not os.path.isfile(csv_file):
        print(f"File not found: {csv_file}")
        return None

    df = pd.read_csv(csv_file, index_col=0)
    if df.empty:
        print(f"No data in file (empty DataFrame): {csv_file}")
        return None

    # Convert index (Y) and columns (X) to float
    df.index = df.index.astype(float)
    df.columns = df.columns.astype(float)

    all_y = df.index.values
    if len(all_y) == 0:
        print(f"No Y data in file: {csv_file}")
        return None

    # Find row in DF closest to y_target
    idx_closest = np.argmin(np.abs(all_y - y_target))
    actual_y = all_y[idx_closest]

    row_data = df.loc[actual_y]  # Pandas Series: index=X, values=U
    return row_data


def main():
    avgfreestreamUc = np.array([])
    avgfreestreamUs = np.array([])
    """
    For each plane i in [1..24],
    1) Extract the entire row at y=3.5 for CC & SC.
    2) Subset X to [120..155].
    3) Convert those X-values to dimensionless X using convert_sx.
    4) Plot dimensionless X vs. U for CC (red) and SC (blue).
    Saves each figure as plane_{i}_u_infinity_dimX.png
    """
    folder = "PIV_planes"
    n_planes = 24

    x_min = 120
    x_max = 155

    for i in range(1, n_planes + 1):
        # Build file paths for CC and SC
        cc_file = os.path.join(folder, f"Case_CC_Span_{i}.txt_u.csv")
        sc_file = os.path.join(folder, f"Case_SC_Span_{i}.txt_u.csv")

        row_cc = extract_u_at_y(cc_file, y_target=3.5)
        row_sc = extract_u_at_y(sc_file, y_target=3.5)

        if row_cc is None or row_sc is None:
            print(f"Skipping plane {i} (missing or empty data).")
            continue

        # 1) Restrict the CC row to x in [120..155]
        #    row_cc.index is an array of X values
        cc_sub = row_cc[(row_cc.index >= x_min) & (row_cc.index <= x_max)]
        # Sort by X for a left-to-right plot
        cc_sub = cc_sub.sort_index()

        # 2) Restrict the SC row to x in [120..155]
        sc_sub = row_sc[(row_sc.index >= x_min) & (row_sc.index <= x_max)]
        sc_sub = sc_sub.sort_index()

        # If either ends up empty, skip
        if cc_sub.empty and sc_sub.empty:
            print(f"Plane {i} has no x in [120..155]. Skipping.")
            continue

        # 3) Convert X to dimensionless
        #    (interpreted as arc-length s for convert_sx)
        cc_dimx = convert_sx(cc_sub.index.values)  # dimensionless X
        sc_dimx = convert_sx(sc_sub.index.values)

        # 4) Prepare for plotting
        u_cc = cc_sub.values  # CC velocity array
        u_sc = sc_sub.values  # SC velocity array

        plt.figure()
        # Plot CC in red, SC in blue
        if len(cc_dimx) > 0:
            plt.plot(cc_dimx, u_cc, label='CC')
        if len(sc_dimx) > 0:
            plt.plot(sc_dimx, u_sc, label='SC')

        avgfreestreamUc = np.append(avgfreestreamUc,np.average(u_cc))
        avgfreestreamUs = np.append(avgfreestreamUs,np.average(u_sc))


        plt.xlabel("Dimensionless X (x/c_x)")
        plt.ylabel("U velocity at y=3.5")
        plt.title(f"Plane {i}: U velocity at y=3.5, X in [120..155] => dimensionless")
        plt.grid(True)
        plt.legend()

        out_name = f"u_infinity_profiles/plane_{i}_u_infinity_dimX.png"
        plt.savefig(out_name, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_name}")
    print(avgfreestreamUc,avgfreestreamUs)
    a = input("Press enter to exit")

if __name__ == "__main__":
    main()

# print('hello')

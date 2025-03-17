#!/usr/bin/env python3

import glob
import os
import pandas as pd
import numpy as np

def compute_u_infinity(csv_file, y_target=3.5):
    """
    Reads the given *_u.csv file, converts row/column labels to float,
    finds the row closest to y_target, and returns the mean (across x)
    of the u velocity at that row.
    """
    # Read the data
    df = pd.read_csv(csv_file, index_col=0)
    # Convert index (Y) and columns (X) to float
    df.index = df.index.astype(float)
    df.columns = df.columns.astype(float)

    # Locate the row in the DataFrame index that is closest to y_target
    all_y = df.index.values
    idx_closest = np.argmin(np.abs(all_y - y_target))
    actual_y = all_y[idx_closest]

    # Extract that row as a Series, then take its mean across columns (X)
    row_data = df.loc[actual_y]
    return row_data.mean()

def compute_u_inf_for_config_span(config, span_number, folder="PIV_planes", y_target=3.5):
    """
    Constructs the filename for a given config ("CC" or "SC") and span_number (1..24).
    E.g.: "Case_CC_Span_1.txt_u.csv"
    Then calculates and returns the average free-stream velocity (u_inf) at y_target.
    """
    # Build the file name from parameters
    file_name = f"Case_{config}_Span_{span_number}.txt_u.csv"
    csv_file = os.path.join(folder, file_name)

    # Use the helper function to compute u_infinity
    u_inf = compute_u_infinity(csv_file, y_target=y_target)
    return u_inf

def main():
    """
    Example usage:
    Loops over span 1..24 for both CC and SC, printing the computed u_infinity.
    """
    for span in range(1, 25):
        u_inf_cc = compute_u_inf_for_config_span("CC", span)
        u_inf_sc = compute_u_inf_for_config_span("SC", span)
        print(f"Span {span}: CC u_inf = {u_inf_cc:.4f}, SC u_inf = {u_inf_sc:.4f}")

if __name__ == "__main__":
    main()
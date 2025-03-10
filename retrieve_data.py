#!/usr/bin/env python3

import os
import pandas as pd

def get_data_sample(case_type, plane_number, field, x_coord, y_coord):
    """
    Retrieve a single data sample from the PIV CSV files.
    
    Parameters:
        case_type   : str, either "clean" or "strips" (mapped to "CC" or "SC").
        plane_number: int, which plane (1 to 25).
        field       : str, one of "u", "v", or "UV".
        x_coord     : float, x-coordinate to retrieve.
        y_coord     : float, y-coordinate to retrieve.
    
    Returns:
        float or None: The requested value at (x_coord, y_coord) if found.
                       Returns None or raises an error if not found.
    """
    # Map "clean" to "CC", "strips" to "SC"
    if case_type.lower() == "clean":
        case_label = "CC"
    elif case_type.lower() == "strips":
        case_label = "SC"
    else:
        raise ValueError('case_type must be either "clean" or "strips".')

    # Build the expected filename. Example: "Case_CC_Span_3.txt_u.csv"
    file_name = f"Case_{case_label}_Span_{plane_number}.txt_{field}.csv"
    
    # Full path to the CSV (adjust if your actual folder structure is different)
    folder_path = "PIV_planes"
    csv_path = os.path.join(folder_path, file_name)
    
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Could not find file: {csv_path}")

    # Read the CSV, first column = index (Y values), columns = X values
    df = pd.read_csv(csv_path, index_col=0)
    
    # Convert column labels (X) to float and row labels (Y) to float
    df.columns = df.columns.astype(float)
    df.index = df.index.astype(float)
    
    # Now we can try to retrieve the specific cell
    # We'll assume x_coord and y_coord exactly match the table's floats
    try:
        value = df.loc[y_coord, x_coord]
        return float(value)
    except KeyError:
        # If exact match wasn't found, handle how you prefer:
        # Option 1: Return None
        # Option 2: Raise an error
        # Below, we'll raise a more descriptive error:
        raise ValueError(
            f"Coordinates (x={x_coord}, y={y_coord}) not found in {csv_path}"
        )

def main():
    # Example usage:
    # Retrieve a single data sample from "clean" category, plane #3, field "UV", at x=117.568086, y=6.923279
    sample = get_data_sample("clean", 3, "UV", 117.568086, 6.923279)
    print("Retrieved Sample:", sample)

if __name__ == "__main__":
    main()

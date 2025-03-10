#!/usr/bin/env python3

import os
import pandas as pd

def get_data_sample(case_type, plane_number, field, x_coord, y_coord):
    """
    Retrieve a single data sample from a CSV file where:
      - The first row (header) contains X-coordinates as columns.
      - The first column (index) contains Y-coordinates as rows.
      
    Parameters:
        case_type   (str) : "clean" (CC) or "strips" (SC).
        plane_number(int) : Which plane (1 through 25).
        field       (str) : "u", "v", or "UV".
        x_coord  (float)  : X coordinate for lookup.
        y_coord  (float)  : Y coordinate for lookup.
        
    Returns:
        float: The requested value from (x_coord, y_coord).
    
    Raises:
        FileNotFoundError: If the expected CSV file does not exist.
        ValueError       : If (x_coord, y_coord) is not found in the file.
    """
    # Convert "clean" -> "CC" or "strips" -> "SC"
    if case_type.lower() == "clean":
        case_label = "CC"
    elif case_type.lower() == "strips":
        case_label = "SC"
    else:
        raise ValueError('Invalid case_type; must be either "clean" or "strips".')
    
    # Build the filename, e.g., "Case_CC_Span_3.txt_v.csv"
    file_name = f"Case_{case_label}_Span_{plane_number}.txt_{field}.csv"
    folder_path = "PIV_planes"  # Adjust if your folder structure differs
    csv_path = os.path.join(folder_path, file_name)
    
    # Ensure the file exists
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    # Read the CSV, with:
    #   - header=0 so that the first row is used for column labels (X coords)
    #   - index_col=0 so that the first column is used for the row index (Y coords)
    df = pd.read_csv(csv_path, header=0, index_col=0)
    
    # Convert the column labels (X coords) to float
    df.columns = df.columns.astype(float)
    # Convert the row labels (Y coords) to float
    df.index = df.index.astype(float)
    
    # Retrieve the requested data cell
    try:
        return float(df.loc[y_coord, x_coord])
    except KeyError:
        raise ValueError(
            f"Coordinates (x={x_coord}, y={y_coord}) not found in {csv_path}."
        )

def main():
    # Example usage:
    # "clean" -> "CC", plane = 1, field = "v",
    # and we want the value at (x=117.189856, y=5.897608).
    try:
        sample = get_data_sample("clean", 2, "v", 117.946316, 7.564323)
        print("Retrieved Sample:", sample)
    except (FileNotFoundError, ValueError) as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
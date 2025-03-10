#!/usr/bin/env python3

import os
import pandas as pd

def get_data_sample(csv_path, x_coord, y_coord):
    """
    Retrieve a single data sample from a CSV file, where:
      - The first row holds X-coordinates (as column headers)
      - The first column holds Y-coordinates (as row indices)
    The interior cells contain the actual data (e.g. v-values).
    
    Parameters:
        csv_path (str): Path to the CSV file
        x_coord (float or int): X coordinate whose data you want
        y_coord (float or int): Y coordinate whose data you want
    
    Returns:
        float: The requested value at (x_coord, y_coord)
    """

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    # Read the CSV
    #   - header=0 tells pandas the first row is column headers
    #   - index_col=0 tells pandas the first column is the row index (Y-coordinates)
    df = pd.read_csv(csv_path, header=0, index_col=0)

    # Convert the column names (X-coordinates) to floats
    df.columns = df.columns.astype(float)
    # Convert the row index (Y-coordinates) to floats
    df.index = df.index.astype(float)

    # Attempt to retrieve the requested data cell
    try:
        return float(df.loc[y_coord, x_coord])
    except KeyError:
        raise ValueError(
            f"Coordinates (x={x_coord}, y={y_coord}) not found in '{csv_path}'."
        )

def main():
    # Example usage: Suppose the CSV is "PIV_planes/Case_CC_Span_1.txt_v.csv"
    # and you want the data at x=117.189856, y=5.897608
    csv_file = "PIV_planes/Case_CC_Span_2.txt_v.csv"
    x_val = 117.946316
    y_val = 7.564323
    
    try:
        sample = get_data_sample(csv_file, x_val, y_val)
        print(f"Data at (x={x_val}, y={y_val}): {sample}")
    except (FileNotFoundError, ValueError) as e:
        print(e)

if __name__ == "__main__":
    main()
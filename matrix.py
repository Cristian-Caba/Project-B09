import numpy as np
import pandas as pd

# Load data
data = pd.read_csv('PIV_planes/Case_CC_Span_1.txt', delim_whitespace=True, skiprows=1)

# Drop NaN rows
data = data.dropna()

# Find unique X values
unique_x = np.unique(data['X'].values)

# Dictionary to hold Y coordinates for each unique X
xy_dict = {}
for x in np.unique(data['X']):
    y_vals = data[data['X'] == x]['Y'].values
    y_vals = np.pad(y_vals, (0, len(data['Y'].unique()) - len(y_vals)), constant_values=np.nan)
    xy_vals = np.array(y_vals)
    xy_vals.sort()
    xy_vals = xy_vals[::-1]
    xy_vals = xy_vals[~np.isnan(xy_vals)]
    xy_vals = np.pad(xy_vals, (0, len(data['Y'].unique()) - len(xy_vals)), constant_values=np.nan)
    if 'matrix' in locals():
        matrix = np.column_stack((matrix, xy_vals))
    else:
        matrix = xy_vals

# Convert to DataFrame for better visibility
matrix_df = pd.DataFrame(matrix, columns=np.unique(data['X']))

# Print the matrix
print(matrix_df)
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# Folder containing all plane files
data_folder = "PIV_planes_dimensionalised"

# List of plane files (Assuming sorted order corresponds to increasing z positions)
plane_files_U = sorted(glob.glob(f"{data_folder}/Case_CC_Span_*_u.csv"))  # U component files
plane_files_V = sorted(glob.glob(f"{data_folder}/Case_CC_Span_*_v.csv"))  # V component files
plane_files_Us = sorted(glob.glob(f"{data_folder}/Case_SC_Span_*_u.csv"))  # U component files
plane_files_Vs = sorted(glob.glob(f"{data_folder}/Case_SC_Span_*_v.csv"))  # V component files

# Lists to store 3D velocity field
U_list, V_list = [], []

# Step 1: Find common X and Y values across all planes
x_sets = []
y_sets = []

for file_U in plane_files_U:
    dfu = pd.read_csv(file_U, index_col=0)
    x_sets.append(set(dfu.columns.astype(float)))
    y_sets.append(set(dfu.index.astype(float)))

for file_V in plane_files_V:
    dfv = pd.read_csv(file_V, index_col=0)
    x_sets.append(set(dfv.columns.astype(float)))
    y_sets.append(set(dfv.index.astype(float)))

for file_Us in plane_files_Us:
    dfus = pd.read_csv(file_Us, index_col=0)
    x_sets.append(set(dfu.columns.astype(float)))
    y_sets.append(set(dfu.index.astype(float)))

for file_Vs in plane_files_Vs:
    dfvs = pd.read_csv(file_Vs, index_col=0)
    x_sets.append(set(dfv.columns.astype(float)))
    y_sets.append(set(dfv.index.astype(float)))

# Define your desired range
x_min, x_max = 0.12, 0.16  # Adjust these values as needed
y_min, y_max = 0, 3.5   # Adjust these values as needed

# Filter and sort the common x and y values within the range
common_x_values = sorted([x for x in set.intersection(*x_sets) if x_min <= x <= x_max])
common_y_values = sorted([y for y in set.intersection(*y_sets) if y_min <= y <= y_max])


# Step 2: Read and stack all planes with consistent shapes
for i in range(len(plane_files_U)):
    dfu = []
    dfv = []
    dfus = []
    dfvs = []
    U_2D = []
    V_2D = []
    dfu = pd.read_csv(plane_files_U[i], index_col=0)
    dfv = pd.read_csv(plane_files_V[i], index_col=0)
    dfus = pd.read_csv(plane_files_Us[i], index_col=0)
    dfvs = pd.read_csv(plane_files_Vs[i], index_col=0)

    # Convert index (Y) and columns (X) to float
    dfu.index = dfu.index.astype(float)
    dfu.columns = dfu.columns.astype(float)
    dfv.index = dfv.index.astype(float)
    dfv.columns = dfv.columns.astype(float)
    dfus.index = dfus.index.astype(float)
    dfus.columns = dfus.columns.astype(float)
    dfvs.index = dfvs.index.astype(float)
    dfvs.columns = dfvs.columns.astype(float)

    # Reindex to ensure consistent shape
    dfu = dfu.reindex(index=common_y_values, columns=common_x_values, fill_value=0)
    dfv = dfv.reindex(index=common_y_values, columns=common_x_values, fill_value=0)
    dfus = dfus.reindex(index=common_y_values, columns=common_x_values, fill_value=0)
    dfvs = dfvs.reindex(index=common_y_values, columns=common_x_values, fill_value=0)

    # Convert DataFrame to NumPy arrays
    U_2D.append(dfus.values-dfu.values)
    V_2D.append(dfvs.values-dfv.values)


    # Get X and Y values from the first plane
    x_values = np.array(common_x_values)
    y_values = np.array(common_y_values)

    X,Y = np.meshgrid(x_values,y_values)

    # Plot settings
    stepx = 15
    stepy = 1

    print(X.shape,Y.shape)

    # Plot the averaged velocity field with centered arrows
    plt.figure(figsize=(10, 10))
    plt.quiver(X[::stepy, ::stepx], Y[::stepy, ::stepx], U_2D[0][::stepy, ::stepx], V_2D[0][::stepy, ::stepx], color="b")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Velocity Field Plot of \n{plane_files_U[i]}")
    plt.grid()


    # Save the figure
    out_image_name = f"{plane_files_U[i]}.csv_difference_velocity_field.png"
    plt.savefig(out_image_name, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_image_name}")
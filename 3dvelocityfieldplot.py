import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

# Folder containing all plane files
data_folder = "PIV_planes"

# List of plane files (Assuming sorted order corresponds to increasing z positions)
plane_files_U = sorted(glob.glob(f"{data_folder}/Case_CC_Span_*_u.csv"))  # U component files
plane_files_V = sorted(glob.glob(f"{data_folder}/Case_CC_Span_*_v.csv"))  # V component files
z_positions = np.linspace(0, len(plane_files_U) - 1, len(plane_files_U))  # Define Z positions

# Lists to store 3D velocity field
U_list, V_list, W_list = [], [], []

# Read and stack all planes
for i, (file_U, file_V) in enumerate(zip(plane_files_U, plane_files_V)):
    dfu = pd.read_csv(file_U, index_col=0)
    dfv = pd.read_csv(file_V, index_col=0)

    # Convert index (Y) and columns (X) to float
    dfu.index = dfu.index.astype(float)
    dfu.columns = dfu.columns.astype(float)
    dfv.index = dfv.index.astype(float)
    dfv.columns = dfv.columns.astype(float)

    # Convert DataFrame to NumPy arrays
    U_list.append(dfu.values)
    V_list.append(dfv.values)
    W_list.append(np.zeros_like(dfu.values))  # Assuming W = 0 unless given

# Stack along the Z dimension
U_3D = np.stack(U_list, axis=-1)  # Shape (Y, X, Z)
V_3D = np.stack(V_list, axis=-1)
W_3D = np.stack(W_list, axis=-1)

# Get X and Y values from the first plane
x_values = dfu.columns.to_numpy()
y_values = dfu.index.to_numpy()

step = 15

plt.figure(figsize=(10, 10))
plt.quiver(x_values[::step], y_values,z_positions, U_3D[:,:,::step], V_3D[:,:,::step],W_3D[:,:,::step], color="b")
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"Velocity Field Plot of \n{base_clean}")
plt.grid()

out_image_name = "3D_velocity_field.png"
out_path = os.path.join(out_image_name)
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {out_path}")
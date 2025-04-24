import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Folder containing all plane files
data_folder = "PIV_planes"

# List of plane files (Assuming sorted order corresponds to increasing z positions)
plane_files_U = sorted(glob.glob(f"{data_folder}/Case_CC_Span_*_u.csv"))  # U component files
plane_files_V = sorted(glob.glob(f"{data_folder}/Case_CC_Span_*_v.csv"))  # V component files
plane_files_Us = sorted(glob.glob(f"{data_folder}/Case_SC_Span_*_u.csv"))  # U component files
plane_files_Vs = sorted(glob.glob(f"{data_folder}/Case_SC_Span_*_v.csv"))  # V component files
z_positions = np.linspace(0, len(plane_files_U) - 1, len(plane_files_U))  # Define Z positions

# Lists to store 3D velocity field
U_list, V_list, W_list, Uc_list, Vc_list = [], [], [], [], []

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
x_min, x_max = 120, 155  # Adjust these values as needed
y_min, y_max = 0, 3.5   # Adjust these values as needed

# Filter and sort the common x and y values within the range
common_x_values = sorted([x for x in set.intersection(*x_sets) if x_min <= x <= x_max])
common_y_values = sorted([y for y in set.intersection(*y_sets) if y_min <= y <= y_max])


# Step 2: Read and stack all planes with consistent shapes
for i, (file_U,file_Us, file_V,file_Vs) in enumerate(zip(plane_files_U,plane_files_Us, plane_files_V,plane_files_Vs)):
    dfu = pd.read_csv(file_U, index_col=0)
    dfv = pd.read_csv(file_V, index_col=0)
    dfus = pd.read_csv(file_Us, index_col=0)
    dfvs = pd.read_csv(file_Vs, index_col=0)

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
    U_list.append(dfus.values) # U_list.append(dfus.values-dfu.values)
    V_list.append(dfvs.values) # V_list.append(dfvs.values-dfv.values)
    W_list.append(np.zeros_like(dfus.values)) # W_list.append((dfus.values-dfu.values)**0.5)  # Assuming W = 0 unless given
    Uc_list.append(dfu.values) # U_list.append(dfus.values-dfu.values)
    Vc_list.append(dfv.values) # V_list.append(dfvs.values-dfv.values)
 
# Stack along the Z dimension
U_3D = np.stack(U_list, axis=-1)  # Shape (Y, X, Z)
V_3D = np.stack(V_list, axis=-1)
Uc_3D = np.stack(Uc_list, axis=-1)  # Shape (Y, X, Z)
Vc_3D = np.stack(Vc_list, axis=-1)
W_3D = np.stack(W_list, axis=-1)


# Get X and Y values from the first plane
x_values = np.array(common_x_values)
y_values = np.array(common_y_values)

Y,X,Z = np.meshgrid(y_values,x_values,z_positions,indexing="ij")

# Plot settings
stepx = 1
stepy = 1


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

#print(np.shape(U_3D[::stepy,::stepx,:]))

listaverageU = np.array([])
listaverageUc = np.array([])
listaverageV = np.array([])
listaverageVc = np.array([])

for i in range(24):
    listaverageU = np.append(listaverageU,np.average(U_3D[:,:,i]))
    listaverageUc = np.append(listaverageUc,np.average(Uc_3D[:,:,i]))
    listaverageV = np.append(listaverageV,np.average(V_3D[:,:,i]))
    listaverageVc = np.append(listaverageVc,np.average(Vc_3D[:,:,i]))

print(np.std(listaverageU))
print(np.std(listaverageUc))
print(np.std(listaverageV))
print(np.std(listaverageVc))

'''
ax.quiver(Y[::stepy,::stepx,:],X[::stepy,::stepx,:], Z[::stepy,::stepx,:], V_3D[::stepy,::stepx,:],U_3D[::stepy,::stepx,:], W_3D[::stepy,::stepx,:],length=0.1)
ax.set_xlabel("Y")
ax.set_ylabel("X")
ax.set_zlabel("Z")
ax.set_title("Velocity Field Plot")
ax.grid()
plt.show()

# Save image
out_image_name = "3D_velocity_field.png"
out_path = os.path.join(out_image_name)
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {out_path}")
'''
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.fft as fft
from matplotlib.ticker import MultipleLocator


# Folder containing all plane files
data_folder = "PIV_planes_dimensionalised"

# List of plane files (Assuming sorted order corresponds to increasing z positions)
plane_files_U = sorted(glob.glob(f"{data_folder}/Case_CC_Span_*.txt_u.csv"))  # U component files
plane_files_V = sorted(glob.glob(f"{data_folder}/Case_CC_Span_*.txt_v.csv"))  # V component files
plane_files_Us = sorted(glob.glob(f"{data_folder}/Case_SC_Span_*.txt_u.csv"))  # U component files
plane_files_Vs = sorted(glob.glob(f"{data_folder}/Case_SC_Span_*.txt_v.csv"))  # V component files
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
x_min, x_max = 0.12, 0.16  # Adjust these values as needed
y_min, y_max = 0, 3.5   # Adjust these values as needed

# Filter and sort the common x and y values within the range
common_x_values = sorted([x for x in set.intersection(*x_sets) if x_min <= x <= x_max])
common_y_values = sorted([y for y in set.intersection(*y_sets) if y_min <= y <= y_max])

avgfreestreamUc = [15.89221041,15.8812731,15.88166506,15.88231474,15.87889637,15.87319347,15.87158631,15.86942512,15.87121722,15.88537732,15.91104011,15.89047006,15.90934316,15.89633224,15.88637055,15.88676133,15.891818,15.88468165,15.88840592,15.89670192,15.89449271,15.89633224,15.89789508,15.89734843] 
avgfreestreamUs = [15.65349674,15.64163651,15.64565217,15.64012264,15.68602373,15.68882426,15.68563513,15.69037647,15.69409897,15.67630015,15.69522935,15.68811139,15.69159944,15.681692,15.67800797,15.69037647,15.72009495,15.7205732,15.72128423,15.7145724,15.71621016,15.72367099,15.72264689,15.72383139]


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

# Y,X,Z = np.meshgrid(y_values,x_values,z_positions,indexing="ij")
Y,Z = np.meshgrid(y_values,z_positions,indexing="ij")

# Plot settings
stepx = 1
stepy = 1


fig = plt.figure(figsize=(10, 10))
#ax = fig.add_subplot(111, projection="3d")

C_3D = np.sqrt(U_3D**2 + V_3D**2)
Cc_3D = np.sqrt(Uc_3D**2 + Vc_3D**2)

DiffV_3D = Vc_3D - V_3D

Diff_3D = Cc_3D - C_3D

vmax = np.max(np.abs(U_3D))  # Max absolute value
vmin = -vmax



#ax.quiver(Y[::stepy,::stepx,:],X[::stepy,::stepx,:], Z[::stepy,::stepx,:], V_3D[::stepy,::stepx,:],U_3D[::stepy,::stepx,:], W_3D[::stepy,::stepx,:],length=0.1)
for x in range(0,len(x_values),15):
    plt.pcolormesh(Z,Y,V_3D[:,x,:]/np.mean(U_3D[27,x,:]),cmap='Spectral') #vmin=vmin,vmax=vmax) # vmin=float(np.min(DiffV_3D)),vmax=float(np.max(DiffV_3D))
    plt.xlabel("z")
    plt.ylabel("y")
    plt.xlim(0, 25)
    plt.ylim(0, 3.5)
    plt.title(f"Velocity Field Plot (SC) at x={x_values[x]}")
    plt.grid()
    plt.colorbar()
    #plt.show()

    # Save image
    out_image_name = f"yzvelocityfield-x{x}.png"
    out_path = f'C:\\Users\\iangh\\Documents\\Python\\GroupB09github\\Project-B09-1\\PIV_planes_dimensionalised\\yz-images\\{out_image_name}'
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")

    # Number of sample points
    N = 24
    # sample spacing
    T = 1


    Uc_3D[:,x,:] = Uc_3D[:,x,:]/np.mean(Uc_3D[27,x,:])
    U_3D[:,x,:] = U_3D[:,x,:]/np.mean(U_3D[27,x,:])
    
    # U_3D[:,x,:] = (U_3D[:,x,:] - np.mean(U_3D[:,x,:]))/np.std(U_3D[:,x,:])

    stdlistCC = np.array([])
    stdlistSC = np.array([])


    for y in range(len(y_values)):
        stdlistCC = np.append(stdlistCC,np.std(Uc_3D[y,x,:]))
        stdlistSC = np.append(stdlistSC,np.std(U_3D[y,x,:]))

        fig, axs = plt.subplots(1)

        yfc = fft.fft(Uc_3D[y,x,:])/24
        xfc = fft.fftfreq(N,T)[:N//2]
        axs.plot(1/xfc,2*N*np.abs(yfc[0:N//2]),marker='o',label='CC')
        yfs = fft.fft(U_3D[y,x,:])/24
        xfs = fft.fftfreq(N,T)[:N//2] 
        axs.plot(1/xfs,2*N*np.abs(yfs[0:N//2]),marker='x',label='SC')
        axs.legend()
        axs.grid()
        axs.set_xlabel('Wavelength [mm]')
        axs.set_ylabel('Amplitude [-]')
        axs.xaxis.set_major_locator(MultipleLocator(1))  # more x ticks
        axs.yaxis.set_major_locator(MultipleLocator(0.1))    # more y ticks
        #axs.set_xticks(np.linspace(0, 24, 25))  # 9 ticks from 1 to 3
        #axs.set_yticks(np.linspace(0, 2, 21))  # 9 ticks from 1 to 9

        plt.suptitle(f"Wave Mode Shapes at x/c = {round(x_values[x],4)} and y = {round(y_values[y],4)}")
        
        plt.show()

        print(1/xfs)
        print(2*N*np.abs(yfs[0:N//2]))



    plt.clf()

    plt.plot(stdlistCC,y_values,label='Clean Config')
    plt.plot(stdlistSC,y_values,label='Strip Config')
    plt.xlabel("Standard Deviation")
    plt.ylabel("y")
    plt.title(f"Standard Deviation at x={x_values[x]}")
    plt.grid()
    plt.legend()
    #plt.show()

    # Save image
    out_image_name = f"standarddeviation-x{x}.png"
    out_path = f'C:\\Users\\iangh\\Documents\\Python\\GroupB09github\\Project-B09-1\\PIV_planes_dimensionalised\\yz-images\\{out_image_name}'
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")



import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
np.set_printoptions(threshold=np.inf)
from pathlib import Path
vmin=9999999.3
vmax=-80000.3
# fig, axes = plt.subplots(figsize=(12, 18))
# axes = axes.flatten()
ccsc="CC"

cmin=-50
cmax=0
for i in range (0,24):
    file_path=Path('Vorticity')/f'Case_{ccsc}_Span_{i+1}.txt_vorticity.csv'
    with open(file_path) as file2:
        matrix=np.genfromtxt(file2, delimiter=',', filling_values=np.nan)
    z_sliced=matrix[1:,1:]
    Z=z_sliced   
    if vmin>np.min(Z):
        vmin=np.min(Z)
    if vmax<np.max(Z):
        vmax=np.max(Z)     

for i in range (0,24):
    file_path=Path('Vorticity')/f'Case_{ccsc}_Span_{i+1}.txt_vorticity.csv'
    with open(file_path) as file2:
        matrix=np.genfromtxt(file2, delimiter=',', filling_values=np.nan)
    x=matrix[0][1:]
    y=matrix[20:,0]
    z_sliced=matrix[20:,1:]
    with open(f'PIV_planes_dimensionalised/Case_{ccsc}_Span_{i+1}.txt_u.csv') as file3:
        xdimension=np.genfromtxt(file3, delimiter=',', filling_values=np.nan)[0][2:]
    nrrow=len(matrix)
    nrcol=len(matrix[1])
    X, Y = np.meshgrid(xdimension, y)
    # Z=z_sliced
    Z = np.clip(z_sliced, cmin, cmax)
    
    
    fig, axes = plt.subplots(figsize=(9, 3))
    c = axes.contourf(X, Y, Z, levels=50, cmap='Spectral',vmin=cmin,vmax=cmax)
    width_dim = 1.4/900
    spacing_dim = 9.2/900
    height_mm = 0.17  # Y in mm

    # Left edge of the first rectangle is x/c=0.125
    rectLeft = [0.125]  # dimensionless

    center0 = rectLeft[0] + width_dim/2
    # Build left edges for rectangles #2..#5
    for k in range(1, 5):
        center_k = center0 + k*spacing_dim
        left_k = center_k - width_dim/2
        rectLeft.append(left_k)

    # Add each rectangle (all black)
    for left in rectLeft:
        rect = patches.Rectangle(
            (left, 0.0),  # bottom-left corner
            width_dim,    # dimensionless width
            height_mm,    # height in mm
            color='black'
        )
        axes.add_patch(rect)
    
    #axes.set_title(f"Plane {i+1}")
    axes.set_xlabel('$x/c_x$')
    axes.set_ylabel('y')
    axes.set_xlim(0.12,0.158)
    axes.set_ylim(0,3)
    levels = np.linspace(cmin, cmax, 10)
    fig.colorbar(c, ax=axes, orientation='vertical', fraction=0.02, pad=0.04,label="ξ [1/s]")
    plt.tight_layout()
    plt.savefig(f"VorticityPlots/Case_{ccsc}_Span_{i+1}.png")
    plt.close(fig)


# plt.tight_layout()

# # Show the plot
# plt.show()

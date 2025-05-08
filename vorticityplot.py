import matplotlib.pyplot as plt
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
    fig, axes = plt.subplots(figsize=(6, 4))
    c = axes.contourf(X, Y, Z, levels=50, cmap='Spectral',vmin=cmin,vmax=cmax)
    axes.set_title(f"Plane {i+1}")
    axes.set_xlabel('Chord')
    axes.set_ylabel('Height')
    axes.set_xlim(0.12,0.158)
    axes.set_ylim(0,3)
    levels = np.linspace(cmin, cmax, 10)
    fig.colorbar(c, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    plt.savefig(f"VorticityPlots/Case_{ccsc}_Span_{i+1}.png")
    plt.close(fig)


# plt.tight_layout()

# # Show the plot
# plt.show()

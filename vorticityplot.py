import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)
from pathlib import Path
vmin=9999999.3
vmax=-80000.3
fig, axes = plt.subplots(4, 6, figsize=(12, 18))
axes = axes.flatten()
ccsc="SC"

cmin=0
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
    with open(f'Project-B09/PIV_planes_dimensionalised/Case_SC_Span_1.txt_u.csv') as file3:
        xdimension=np.genfromtxt(file3, delimiter=',', filling_values=np.nan)[0][1:0]
    nrrow=len(matrix)
    nrcol=len(matrix[1])
    X, Y = np.meshgrid(xdimension, y)
    Z=z_sliced
    c = axes[i].contourf(X, Y, Z, levels=50, cmap='jet',vmin=cmin,vmax=cmax)
    axes[i].set_title(f"Frame{i+1}")
    axes[i].set_xlabel('Chord')
    axes[i].set_ylabel('Height')
    axes[i].set_xlim(120,140)
    axes[i].set_ylim(0,3)
plt.figure
levels = np.linspace(vmin, vmax, 10)

fig.colorbar(c, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
plt.tight_layout()

# Show the plot
plt.show()

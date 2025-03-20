import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)
from pathlib import Path
vmin=9999999.3
vmax=-80000.3
fig, axes = plt.subplots(4, 6, figsize=(12, 18))
axes = axes.flatten()
file_path=Path('Vorticity')/f'Case_CC_Span_{1}.txt_vorticity.csv'
with open(file_path) as file2:
    matrix=np.genfromtxt(file2, delimiter=',', filling_values=np.nan)
x=matrix[0][1:]
y=matrix[1:,0]
z_sliced=matrix[1:,1:]
vmin=min(vmin,np.min(z_sliced))
vmax=max(vmax,np.max(z_sliced))

for i in range (0,24):
    file_path=Path('Vorticity')/f'Case_CC_Span_{i+1}.txt_vorticity.csv'
    with open(file_path) as file2:
        matrix=np.genfromtxt(file2, delimiter=',', filling_values=np.nan)
    x=matrix[0][1:]
    y=matrix[1:,0]
    z_sliced=matrix[1:,1:]
    
    nrrow=len(matrix)
    nrcol=len(matrix[1])
    X, Y = np.meshgrid(x, y)
    Z=z_sliced
    c = axes[i].contourf(X, Y, Z, levels=50, cmap='jet',vmin=vmin,vmax=vmax)
    axes[i].set_title(f"Frame{i+1}")
    axes[i].set_xlabel('Chord')
    axes[i].set_ylabel('Height')
plt.figure
levels = np.linspace(vmin, vmax, 10)

fig.colorbar(c, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
plt.tight_layout()

# Show the plot
plt.show()
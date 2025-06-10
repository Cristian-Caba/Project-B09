import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from pathlib import Path
import matplotlib.gridspec as gridspec  # Import gridspec

# Fixed color range for all plots
cmin = -50
cmax = 0

# Create a single normalization object for consistent coloring
norm = plt.Normalize(vmin=cmin, vmax=cmax)
levels = np.linspace(cmin, cmax, 50)

# Define rectangle parameters (same for both configurations)
width_dim = 1.4/900
spacing_dim = 9.2/900
height_mm = 0.17
rectLeft = [0.125]  # dimensionless
center0 = rectLeft[0] + width_dim/2
for k in range(1, 5):
    center_k = center0 + k*spacing_dim
    rectLeft.append(center_k - width_dim/2)

for i in range(0, 24):
    # Create figure with GridSpec layout
    fig = plt.figure(figsize=(14, 3.5))  # Slightly wider figure
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])  # 3 columns: SC, CC, colorbar
    
    ax1 = plt.subplot(gs[0])  # SC plot
    ax2 = plt.subplot(gs[1])  # CC plot
    cax = plt.subplot(gs[2])  # Colorbar axis
    
    # Process SC configuration
    ccsc = "SC"
    file_path = Path('Vorticity')/f'Case_{ccsc}_Span_{i+1}.txt_vorticity.csv'
    with open(file_path) as file2:
        matrix = np.genfromtxt(file2, delimiter=',', filling_values=np.nan)
    x = matrix[0][1:]
    y = matrix[20:,0]
    z_sliced = matrix[20:,1:]
    with open(f'PIV_planes_dimensionalised/Case_{ccsc}_Span_{i+1}.txt_u.csv') as file3:
        xdimension = np.genfromtxt(file3, delimiter=',', filling_values=np.nan)[0][2:]
    X, Y = np.meshgrid(xdimension, y)
    Z = np.clip(z_sliced, cmin, cmax)
    
    # Plot SC
    c1 = ax1.contourf(X, Y, Z, levels=levels, cmap='Spectral', norm=norm)
    ax1.set_title(f"SC Configuration - Plane {i+1}")
    ax1.set_xlabel('$x/c_x$')
    ax1.set_ylabel('y [mm]')
    ax1.set_xlim(0.12, 0.158)
    ax1.set_ylim(0, 3)
    
    # Add rectangles for SC
    for left in rectLeft:
        rect = patches.Rectangle(
            (left, 0.0), width_dim, height_mm, color='black'
        )
        ax1.add_patch(rect)
    
    # Process CC configuration
    ccsc = "CC"
    file_path = Path('Vorticity')/f'Case_{ccsc}_Span_{i+1}.txt_vorticity.csv'
    with open(file_path) as file2:
        matrix = np.genfromtxt(file2, delimiter=',', filling_values=np.nan)
    x = matrix[0][1:]
    y = matrix[20:,0]
    z_sliced = matrix[20:,1:]
    with open(f'PIV_planes_dimensionalised/Case_{ccsc}_Span_{i+1}.txt_u.csv') as file3:
        xdimension = np.genfromtxt(file3, delimiter=',', filling_values=np.nan)[0][2:]
    X, Y = np.meshgrid(xdimension, y)
    Z = np.clip(z_sliced, cmin, cmax)
    
    # Plot CC
    c2 = ax2.contourf(X, Y, Z, levels=levels, cmap='Spectral', norm=norm)
    ax2.set_title(f"CC Configuration - Plane {i+1}")
    ax2.set_xlabel('$x/c_x$')
    ax2.set_xlim(0.12, 0.158)
    ax2.set_ylim(0, 3)
    
    # Add shared colorbar in its dedicated axis
    cbar = fig.colorbar(c1, cax=cax, label="ξ [1/s]")
    cbar.set_ticks(np.arange(-50, 10, 10))  # Consistent ticks: -50, -40, ..., 0
    
    plt.tight_layout()
    plt.savefig(f"VorticityPlots/Span_{i+1}_SC_CC.png", bbox_inches='tight')
    plt.close(fig)
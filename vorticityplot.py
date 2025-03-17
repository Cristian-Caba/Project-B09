import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)
from pathlib import Path
for i in range (7,8):
    file_path=Path('Project-B09/Vorticity')/f'Case_SC_Span_{i}.txt_vorticity.csv'
    with open(file_path) as file2:
        matrix=np.genfromtxt(file2, delimiter=',', filling_values=np.nan)
    x=matrix[0][15:280]
    y=matrix[30:,0]
    z_sliced=matrix[230:,15:280]
    nrrow=len(matrix)
    nrcol=len(matrix[1])
    X, Y = np.meshgrid(x, y)
Z=z_sliced
plt.figure
levels = np.linspace(np.min(Z), np.max(Z), 10)
plt.contourf(X, Y, Z, levels=20,cmap='jet')
plt.colorbar(label="DIng")  # Show color bar
plt.xlabel("X-axis (excluding first row)")
plt.ylabel("Y-axis (excluding first column)")
plt.title("🔥 Contour Plot without First Row and Column 🔥")
plt.show()
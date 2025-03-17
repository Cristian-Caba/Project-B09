from pathlib import Path
import numpy as np
np.set_printoptions(threshold=np.inf)
def vorticity(umatrix,vmatrix):
    nrrow=len(umatrix)
    nrcol=len(umatrix[1])
    vormatrix=np.full((nrrow-1, nrcol-1), np.nan)
    for i in range(2,nrrow):
        for j in range(2,nrcol):
            vormatrix[0][j-1]=umatrix[0][j]
            vormatrix[i-1][0]=umatrix[i][0]
            dx=umatrix[0][j]-umatrix[0][j-1]
            dy=umatrix[i][0]-umatrix[i-1][0]
            du=umatrix[i][j]-umatrix[i-1][j]
            dv=vmatrix[i][j]-vmatrix[i][j-1]
            val=dv/dx-du/dy
            vormatrix[i-1][j-1]=val
    return vormatrix

""" folder_path = "PIV_planes"
file_patterns = ["*_u.csv", "*_v.csv", "*_UV.csv"] """


for i in range(1,25):
    file_path = Path('Project-B09/PIV_planes') / f'Case_CC_Span_{i}.txt_u.csv'
    with open(file_path) as file2:
        umatrix=np.genfromtxt(file2, delimiter=',', filling_values=np.nan)
    file_path = Path('Project-B09/PIV_planes') / f'Case_CC_Span_{i}.txt_v.csv'
    with open(file_path) as file2:
        vmatrix=np.genfromtxt(file2, delimiter=',', filling_values=np.nan)
    np.savetxt(f'Project-B09/Vorticity/Case_CC_Span_{i}.txt_vorticity.csv', vorticity(umatrix,vmatrix), delimiter=",", fmt="%.4f")
    file_path = Path('Project-B09/PIV_planes') / f'Case_SC_Span_{i}.txt_u.csv'
    with open(file_path) as file2:
        umatrix=np.genfromtxt(file2, delimiter=',', filling_values=np.nan)
    file_path = Path('Project-B09/PIV_planes') / f'Case_SC_Span_{i}.txt_v.csv'
    with open(file_path) as file2:
        vmatrix=np.genfromtxt(file2, delimiter=',', filling_values=np.nan)
    np.savetxt(f'Project-B09/Vorticity/Case_SC_Span_{i}.txt_vorticity.csv', vorticity(umatrix,vmatrix), delimiter=",", fmt="%.4f")
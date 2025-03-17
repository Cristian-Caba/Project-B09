from pathlib import Path
import numpy as np
def vorticity(umatrix,vmatrix):
    nrrow=len(umatrix)
    nrcol=len(umatrix[0])
    vormatrix=np.full((3, 3), np.nan)
    for i in range(2,nrcol):
        for j in range(2,nrrow):
            vormatrix[0][j-1]=umatrix[0][j]
            vormatrix[i-1][0]=umatrix[i][0]
            dx=umatrix[0][j]-umatrix[0][j-1]
            dy=umatrix[i][0]-umatrix[i-1][0]
            du=umatrix[i][j]-umatrix[i-1][j]
            dv=vmatrix[i][j]-vmatrix[i][j-1]
            val=dv/dx-du/dy
            vormatrix[i-1][j-1].append(val)
    return vormatrix

""" folder_path = "PIV_planes"
file_patterns = ["*_u.csv", "*_v.csv", "*_UV.csv"] """


for i in range(1,25):
    file_path = Path('Project-B09/PIV_planes') / f'Case_CC_Span_{i}.txt_u.csv'
    with open(file_path) as file:
        umatrix=np.loadtxt(file, delimiter=',', skiprows=1)
        print (umatrix)


 
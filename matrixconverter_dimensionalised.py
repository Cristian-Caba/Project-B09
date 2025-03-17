import pandas as pd
import math
import numpy as np
from pathlib import Path
from dimensionalization import convert_sx

folder_path = Path('PIV_planes/')

for file_path in folder_path.glob('*.txt'):  # For CSV files
    df = pd.read_csv(file_path, delimiter=" ",skiprows=1)
    a=0

    #columnindex = np.array([])
    columnindex = []
    rowindex = []

    for i in range(60):
        rowindex.append(0)
   

    for i in range(len(df)):
        if df.at[i,"X"] != a:
            a = df.at[i,"X"]
            columnindex.append(df.at[i,"X"])
            #np.append(columnindex,a)
        if 0 < i < 61:
            rowindex[-i] = (df.at[i-1,"Y"])
        else:
            continue
    
    columns_array = np.array(columnindex)

    matrixu = np.zeros((len(rowindex),len(columnindex)))

    matrixv = np.zeros((len(rowindex),len(columnindex)))

    matrixUV = np.zeros((len(rowindex),len(columnindex)))

    normalisedcolumns = list(convert_sx(columns_array))

    






    datamatrixu = pd.DataFrame(matrixu, index=rowindex, columns=columnindex)

    datamatrixv = pd.DataFrame(matrixv, index=rowindex, columns=columnindex)

    datamatrixUV = pd.DataFrame(matrixUV, index=rowindex, columns=columnindex)


    for i in columnindex:
        for j in rowindex:
            u = df[(df['X'] == i) & (df['Y'] == j)]["U"].values[0]
            v = df[(df['X'] == i) & (df['Y'] == j)]["V"].values[0]
            if math.isnan(u) or math.isnan(v):
                datamatrixu.at[j,i] = 0
                datamatrixv.at[j,i] = 0
                datamatrixUV.at[j,i] = 0
                if math.isnan(u) is False:
                    datamatrixu.at[j,i] = u
                elif math.isnan(v) is False:
                    datamatrixv.at[j,i] = v
            else:
                datamatrixu.at[j,i] = u
                datamatrixv.at[j,i] = v
                datamatrixUV.at[j,i] = math.sqrt(u**2+v**2)
            #print(datamatrixu.at[j,i],datamatrixv.at[j,i])

    datamatrixu.columns = normalisedcolumns
    datamatrixv.columns = normalisedcolumns
    datamatrixUV.columns = normalisedcolumns

    print(datamatrixUV)

    file_pathu = str(file_path) + "_u.csv"

    file_pathv = str(file_path) + "_v.csv"

    file_pathUV = str(file_path) + "_UV.csv"

    datamatrixu.to_csv(file_pathu, index=True)
    datamatrixv.to_csv(file_pathv, index=True)
    datamatrixUV.to_csv(file_pathUV, index=True)

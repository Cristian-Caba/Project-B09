import pandas as pd
import numpy as np
from pathlib import Path

folder_path = Path('DataSet_ready/PIV_planes/')

for file_path in folder_path.glob('*.txt'):  # For CSV files
    df = pd.read_csv(file_path, delimiter=" ",skiprows=1)
    a=0

    columnindex = []
    rowindex = []

    for i in range(60):
        rowindex.append(0)


    for i in range(len(df)):
        if df.at[i,"X"] != a:
            a = df.at[i,"X"]
            columnindex.append(df.at[i,"X"])
        if 0 < i < 61:
            rowindex[-i] = (df.at[i-1,"Y"])
        else:
            continue

    matrix = np.zeros((len(rowindex),len(columnindex)))



    datamatrix = pd.DataFrame(matrix, index=rowindex, columns=columnindex)


    for i in columnindex:
        for j in rowindex:
            u = df[(df['X'] == i) & (df['Y'] == j)]["U"].values[0]
            v = df[(df['X'] == i) & (df['Y'] == j)]["V"].values[0]
            if u == "NaN" or v == "NaN":
                matrix[j,i] = 0
            datamatrix.at[j,i] = math.sqrt(u**2+v**2)
            print(df[(df['X'] == i) & (df['Y'] == j)]["U"].values[0])



    print(datamatrix)

    file_path = str(file_path) + ".csv"

    datamatrix.to_csv(file_path, index=True)

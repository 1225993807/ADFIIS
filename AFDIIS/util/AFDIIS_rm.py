import copy
import numpy as np
from scipy.spatial.distance import cdist


def AFDIIS_rm(subdata0, e, category, MISSING_DATA):
    # category 1*m 代表每列的类型 1 为数值，2为布尔，3为类别
    subdata = copy.deepcopy(subdata0)
    if np.ndim(subdata) == 1:
        subdata = np.expand_dims(subdata, np.ndim(subdata))
    e = np.expand_dims(e, np.ndim(e))

    t = subdata[:, 0]
    if np.ndim(t) == 1:
        t = np.expand_dims(t, np.ndim(t))
    if category[0] != 1:
        # 标称
        rm = distance_matrix_bool(t, MISSING_DATA=MISSING_DATA)
    else:
        # 数值
        rm = distance_matrix_numeric(t, MISSING_DATA, e[0])
    for j in range(1, len(e)):
        t = subdata[:, j]
        if np.ndim(t) == 1:
            t = np.expand_dims(t, np.ndim(t))
        if category[j] != 1:
            # 标称
            temp = distance_matrix_bool(t, MISSING_DATA=MISSING_DATA)
        else:
            # 数值
            temp = distance_matrix_numeric(t, MISSING_DATA, e[j])
        rm = np.minimum(rm, temp)

    return rm


def distance_matrix_bool(A, MISSING_DATA):
    n = len(A)
    D = np.zeros((n, n))
    P = 0
    column = []
    column_index = []
    for i in range(n):
        if str(A[i][0]) != MISSING_DATA:
            column.append(A[i][0])
        else:
            column_index.append(i)
            A[i][0] = 0
    unique_values = np.unique(np.array(column))
    P += len(unique_values)
    temp = cdist(np.array(A).astype(float), np.array(A).astype(float), metric='cityblock')
    temp[temp > 0] = 1
    D = 1 - temp
    for index in column_index:
        # 行
        for j in range(n):
            D[index, j] = 0
        # 列
        for i in range(n):
            D[i, index] = 0
    for i in range(n):
        D[i, i] = 1
    return D


def distance_matrix_numeric(A, MISSING_DATA, e):
    n = len(A)
    column_index = []
    D = np.zeros((n, n))
    for i in range(n):
        if str(A[i][0]) == MISSING_DATA:
            A[i][0] = 0
            column_index.append(i)
    temp = cdist(np.array(A).astype(float), np.array(A).astype(float), metric='cityblock')
    temp[temp > e] = 1
    D = 1 - temp
    for index in column_index:
        # 行
        for j in range(n):
            D[index, j] = 1
            # 列
        for i in range(n):
            D[i, index] = 1
    for i in range(n):
        D[i, i] = 1
    return D


if __name__ == '__main__':
    trandata = np.array(
        [[2, 0.8, 0.33333],
         [0, 0.2, 0.66667],
         [2, 0.6, 0],
         [0, '*', 0.11111],
         [0, 0.4, 0.77778],
         [1, 1, 1]]
    )
    print(AFDIIS_rm(trandata, [0, 0.37417, 0.39545], [3, 1, 1], '*'))

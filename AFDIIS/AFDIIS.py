import numpy as np
from util.AFDIIS_rm import AFDIIS_rm
from util.categoryDef import categoryDef
from util.encode_column import encode_column
from util.normalize_column import normalize_column


def AFDIIS(data, delta, category, MISSING_DATA):
    # category 1*m 代表每列的类型 1 为数值，2为布尔，3为类别
    n, m = data.shape
    varepsilon = np.zeros(m)

    for j in range(m):
        tmp = [float(e) for e in data[:, j] if e != MISSING_DATA]
        if category[j] == 1:
            varepsilon[j] = np.std(tmp, ddof=1) / delta

    E = np.zeros(m)
    for j in range(m):
        r1 = AFDIIS_rm(data[:, j], varepsilon[j], [category[j]], MISSING_DATA)
        # 补熵
        E[j] = 1 - (1 / n) * (np.sum(np.sum(r1, axis=1) / n))
    b_de = np.argsort(E)[::-1]
    b_as = np.argsort(E)

    weight = np.zeros((n, m))
    FG_de = np.zeros((n, m))
    FG_as = np.zeros((n, m))

    for k in range(m):
        FSet = AFDIIS_rm(data[:, k], varepsilon[k], [category[k]], MISSING_DATA)
        FSet_de = AFDIIS_rm(data[:, b_de[:(m - k)]], varepsilon[b_de[:(m - k)]], category[b_de[:(m - k)]], MISSING_DATA)
        FSet_as = AFDIIS_rm(data[:, b_as[:(m - k)]], varepsilon[b_as[:(m - k)]], category[b_as[:(m - k)]], MISSING_DATA)
        for i in range(n):
            weight[i, k] = np.sqrt(np.sum(FSet[i, :]) / n)
            FG_de[i, k] = np.sum(FSet_de[i, :])
            FG_as[i, k] = np.sum(FSet_as[i, :])

    FGS_de = np.zeros((n, m - 1))
    FGS_as = np.zeros((n, m - 1))

    for k in range(1, m):
        # 相对差异
        FGS_de_temp = (FG_de[:, k] - FG_de[:, k - 1]) / (FG_de[:, k] + FG_de[:, k - 1])
        FGS_as_temp = (FG_as[:, k] - FG_as[:, k - 1]) / (FG_as[:, k] + FG_as[:, k - 1])
        FGS_de[:len(FGS_de_temp), k - 1] = FGS_de_temp
        FGS_as[:len(FGS_as_temp), k - 1] = FGS_as_temp

    res = np.zeros(n)
    for j in range(n):
        res[j] = 1 - np.sum(weight[j, :]) / m * np.sqrt(
            (np.sum(FGS_de[j, :]) + np.sum(FGS_as[j, :])) / (2 * m - 2))

    return res


if __name__ == '__main__':

    trandata = np.array([['c', 4.0000, 0.7000],
                         ['a', "*", 0.4000],
                         ['c', 1.0000, 0.6000],
                         ['*', 2.0000, 0.3000],
                         ['a', 8.0000, 0.5000],
                         ['b', 10.0000, "*"]])
    category = np.array(categoryDef(trandata, '*'))
    l = np.where(category == 1)[0].tolist()
    n, m = trandata.shape
    for i in range(m):
        if i in l:
            trandata[:, i] = normalize_column(trandata[:, i], '*')
        else:
            trandata[:, i] = encode_column(trandata[:, i], '*')
    sigma = 1
    out_scores = AFDIIS(trandata, sigma, category, '*')
    print(out_scores)

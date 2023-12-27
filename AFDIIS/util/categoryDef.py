def count_unique_elements(column):
    unique_elements = set()

    for element in column:
        if element != '*':
            unique_elements.add(element)

    return unique_elements

def categoryDef(data,MISSING_DATA):
    # category 代表每列的类型 1 为数值，2为布尔，3为类别
    res = []
    n, m = data.shape
    for i in range(m):
        try:
            k = 0
            while str(data[k, i]) == MISSING_DATA:
                k += 1
            value = float(data[k, i])
            res.append(1)
        except ValueError:
            tmp = count_unique_elements(data[:, i])
            if len(tmp) == 2 and (('t' in tmp and 'f' in tmp) or ('T' in tmp and 'F' in tmp)):
                res.append(2)
            else:
                res.append(3)
    return res

from sklearn.preprocessing import LabelEncoder


def encode_column(column, MISSING_DATA):
    encoder = LabelEncoder()
    encoder.fit([element for element in column if element != MISSING_DATA])

    for i in range(len(column)):
        if column[i] != MISSING_DATA:
            column[i] = encoder.transform([column[i]])[0]
    return column

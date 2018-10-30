def multiply(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    if cols_A != rows_B:
        print("Cannot multifply tow matrices. Incorrect dimensions.")
        return

    result = [[0 for row in range(cols_B)] for col in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    
    return result

A = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
B = [[2, 3, 4, 5, 6], [4, 5, 6, 7, 8], [6, 7, 8, 9, 10]]
result = multiply(A, B)
print(result)
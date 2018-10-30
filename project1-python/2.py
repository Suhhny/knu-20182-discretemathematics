import numpy as np

A_data = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
B_data = np.array([[2, 3, 4, 5, 6], [4, 5, 6, 7, 8], [6, 7, 8, 9, 10]])

# result = np.zeros(4, 5) # generate 4 * 5 matrix, 인수 0

result = np.ones((4, 5), dtype="i8") # generate 4 * 5 materix, 인수 1
print(result)

result = np.dot(A_data, B_data)   # result = A_data * B_data로 갱신
print(result)
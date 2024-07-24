import numpy as np

# Create an array
array = np.array([1, 2, 3, 4, 5])

#Basic ops
sum_array = np.sum(array)
mean_array = np.mean(array)
print(f"Sum: {sum_array}, Mean: {mean_array}")

# Creating a 2d array
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix)

# martix ops
transpose = np.transpose(matrix)
print(transpose)

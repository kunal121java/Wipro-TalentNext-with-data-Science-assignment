import numpy as np

# Exercise 1: Create two dimensional, 3*3 array and perform ndim, shape, slicing
# operation on it.

# Create a 3x3 NumPy array
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Print the array
print("Original 3x3 Array:")
print(array_2d)
print("-" * 20)

# 1. ndim: Get the number of dimensions
print("Number of dimensions (ndim):", array_2d.ndim)
print("-" * 20)

# 2. shape: Get the dimensions of the array (rows, columns)
print("Shape of the array (shape):", array_2d.shape)
print("-" * 20)

# 3. Slicing: Extract specific elements or subarrays
# Get the element at the second row, third column
print("Element at [1, 2]:", array_2d[1, 2])

# Get the entire first row
print("First row:", array_2d[0, :])

# Get the first two rows and all columns
print("First two rows:\n", array_2d[:2, :])



# Exercise 2: Create one dimensional array and perform ndim, shape, reshape
# operation on it.

# Create a one-dimensional NumPy array
array_1d = np.array([10, 20, 30, 40, 50, 60])

# Print the array
print("Original 1D Array:", array_1d)
print("-" * 20)

# 1. ndim: Get the number of dimensions
print("Number of dimensions (ndim):", array_1d.ndim)
print("-" * 20)

# 2. shape: Get the dimensions of the array (number of elements)
print("Shape of the array (shape):", array_1d.shape)
print("-" * 20)

# 3. reshape: Change the shape of the array (e.g., to a 2x3 matrix)
# The total number of elements must remain the same (6 elements here)
reshaped_array = array_1d.reshape(2, 3)
print("Reshaped into a 2x3 matrix:\n", reshaped_array)
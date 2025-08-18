# Problem: Write a program to cube every number in a given list.
# The given list is: list_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
list_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Use a lambda function with map() to cube each number
cubed_list = list(map(lambda x: x**3, list_1))

print(cubed_list)
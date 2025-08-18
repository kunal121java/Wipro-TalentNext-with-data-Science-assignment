# Question: Write an LC program to create an output dictionary which contains only
# the odd numbers that are present in the input list = [1,2,3,4,5,6,7] as
# keys and their cubes as values.

input_list = [1, 2, 3, 4, 5, 6, 7]

# Dictionary comprehension to filter for odd numbers and cube them
output_dict = {num: num**3 for num in input_list if num % 2 != 0}

print(output_dict)

# Question: Make a dictionary of the 26 english alphabets mapping each with the
# corresponding integer.

# Import the string module to easily get the lowercase alphabets
import string

# Dictionary comprehension to map each alphabet to its index (plus 1)
alphabet_map = {alphabet: index + 1 for index, alphabet in enumerate(string.ascii_lowercase)}

print(alphabet_map)
# 1. Write a program to accept two numbers from the user and perform division.
#    If any exception occurs, print an error message or else print the result.

try:
    num1 = float(input("Enter first number: "))
    num2 = float(input("Enter second number: "))
    result = num1 / num2
    print("Result:", result)
except Exception as e:
    print("Error:", e)


# 2. Write a program to accept a number from the user and check whether it's prime or not.
#    If user enters anything other than number, handle the exception and print an error message.

try:
    num = int(input("Enter a number: "))
    if num < 2:
        print(f"{num} is not a prime number.")
    else:
        for i in range(2, int(num**0.5)+1):
            if num % i == 0:
                print(f"{num} is not a prime number.")
                break
        else:
            print(f"{num} is a prime number.")
except Exception as e:
    print("Error:", e)


# 3. Write a program to accept the file name to be opened from the user,
#    if file exists print the contents of the file in title case,
#    else handle the exception and print an error message.

try:
    filename = input("Enter file name: ")
    with open(filename, 'r') as file:
        content = file.read()
        print(content.title())
except Exception as e:
    print("Error:", e)


# 4. Declare a list with 10 integers and ask the user to enter an index.
#    Check whether the number in that index is positive or negative.
#    If any invalid index is entered, handle the exception and print an error message.

numbers = [5, -3, 12, -1, 0, 25, -7, 8, -9, 4]
try:
    index = int(input("Enter index (0-9): "))
    num = numbers[index]
    if num > 0:
        print(f"Number at index {index} is positive.")
    elif num < 0:
        print(f"Number at index {index} is negative.")
    else:
        print(f"Number at index {index} is zero.")
except Exception as e:
    print("Error:", e)
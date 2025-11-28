def calculate_factorial(n):

    
    if n < 0:
        return "Factorial is not defined for negative numbers."
    elif n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result


number = 5
print(f"The factorial of {number} is: {calculate_factorial(number)}")


number = 7
print(f"The factorial of {number} is: {calculate_factorial(number)}")
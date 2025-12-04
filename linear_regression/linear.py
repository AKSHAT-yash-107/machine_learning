# Linear Regression using pure maths (no libraries)

# 1) Take number of points
n = int(input("How many data points? "))

x_vals = []
y_vals = []

# 2) Take input points
for i in range(n):
    print(f"Point {i+1}:")
    x = float(input("  x = "))
    y = float(input("  y = "))
    x_vals.append(x)
    y_vals.append(y)

# 3) Compute required sums
sum_x = 0
sum_y = 0
sum_xy = 0
sum_x2 = 0

for i in range(n):
    sum_x += x_vals[i]
    sum_y += y_vals[i]
    sum_xy += x_vals[i] * y_vals[i]
    sum_x2 += x_vals[i] * x_vals[i]

# 4) Apply formula for slope (m) and intercept (c)
# m = ( n*Σxy - Σx*Σy ) / ( n*Σx² - (Σx)² )
numerator_m = n * sum_xy - (sum_x * sum_y)
denominator_m = n * sum_x2 - (sum_x * sum_x)

if denominator_m == 0:
    print("Cannot fit a unique line (denominator is zero).")
else:
    m = numerator_m / denominator_m
    c = (sum_y - m * sum_x) / n

    print("\n=== Linear Regression Result ===")
    print(f"Slope (m)     = {m}")
    print(f"Intercept (c) = {c}")
    print(f"Equation: y = {m} * x + {c}")

    # 5) Predict for a new x
    x_new = float(input("\nEnter x to predict y: "))
    y_pred = m * x_new + c
    print(f"Predicted y for x = {x_new} is {y_pred}")

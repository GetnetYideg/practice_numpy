import numpy as np

# Study hours
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Exam scores
y = np.array([45, 50, 55, 65, 70, 75, 85, 90])

print("Features: ", x)
print("Targets: ", y)

w = 5
b = 40

y_pred = x * w + b
print("Predicted value: ", y_pred)

pred_errors = y -y_pred
print("Errors: ", pred_errors)

# Mean squared error
mse = np.mean((y - y_pred) **2)
print("MSE: ", mse)

## w -> describes how much times y changes when x increases
## b -> describes the initial y value
## y_pred describes the pridicted values of y.
## MSE tells us the magnitude error of a predicted value from the actual value
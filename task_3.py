import numpy as np

# Study hours
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Exam scores
y = np.array([45, 50, 55, 65, 70, 75, 85, 90])

print("Features: ", x)
print("Targets: ", y)

def predict_display_y (w, b):
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

predict_display_y(10, 40)
predict_display_y(5, 50)
predict_display_y(6, 35)

## when the prediction slop (m) increases the prediction value also increases and exceeds the actual value
## when the bias (b) increases the prediction value also increases and it may miss the first value
## the third trial has lowest MSE model.
## It describes mostly the magnitude of an error from the actual value and it is one attribute to perfect the prediction but not gurenties it.

def predict_y (w, b):
    y_pred = x * w + b
    return y_pred

print(predict_y(5, 40)[0], "\t", predict_y(5, 40)[-1])
print(predict_y(5, 50)[0], "\t", predict_y(5, 50)[-1])
print(predict_y(10, 40)[0], "\t", predict_y(10, 40)[-1])

## the value of w changes the slope 
## the value of b shifts the line up or down 
## two different models can have the same slope but different predictions due to bias value defference

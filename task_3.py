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

def compute_dw (y_pred):
    dw = np.mean(x * (y_pred - y)) * 2
    return dw

def compute_db (y_pred):
    db = np.mean(y_pred - y) * 2
    return db

def compute_mse (y_pred):
    mse = np.mean((y - y_pred) **2)
    return mse

rate = 0.001
def update_slope_bias (w, b):

    dw = compute_dw(predict_y(w, b))
    new_w = w - (rate * dw)

    db = compute_db(predict_y(w, b))
    new_b = b - (rate * db)

    return new_w, new_b

w = 0
b = 0
new_w, new_b = update_slope_bias(w, b)
print("Old value of slope: ", w, "\nNew value of slope: ", new_w)
print("Old value of bias: ", b, "\nNew value of bias: ", new_b)

def train_1000 (w, b):
    for i in range(1000): 
        y_pred = predict_y(w, b)
        w, b = update_slope_bias(w, b)

        if i % 100 == 0:
            print("Epoch:", i, "MSE:", compute_mse(y_pred))
    return w, b

final_w, final_b = train_1000(w, b)

print("Final weight:", final_w)
print("Final bias:", final_b)

final_predictions = predict_y(final_w, final_b)
print("Final MSE:", compute_mse(final_predictions))

## Visualize the learning process
print("Actual scores: ", y)
print("Predicted scores: ", final_predictions)

final_errors = y - final_predictions
print("Prediction errors: ", final_errors)

## The predictions on w = 5 and b = 40 is closer but not exact
## gradient descent updates w and b repeatedly to get a closer prediction
## A learning rate that's too large can cause:
# - The model to jump over the minimum error point.
# - The error to increase instead of decrease.
# - Training to become unstable or even diverge.
import numpy as np

data = np.array([
    [50,  1, 20, 8, 120],
    [65,  2, 15, 6, 150],
    [80,  2, 10, 5, 190],
    [90,  3,  8, 4, 220],
    [100, 3,  5, 3, 260],
    [120, 4,  4, 2, 310],
    [150, 4,  2, 1, 380],
    [70,  2, np.nan, 7, 165],
    [110, np.nan, 6, 3, 280],
    [130, 4,  3, np.nan, 340]
])

# Understanding the dataset
## Shape of the dataset
print("Shape: ", data.shape)

## Number of house
print("Number of house: ", data.shape[0])

## Number of features
print("Number of features: ", data.shape[1])

## Extract all columns sperately
col_1 = data[:, 0:1]
col_2 = data[:, 1:2]
col_3 = data[:, 2:3]
col_4 = data[:, 3:4]
col_5 = data[:, 4:5]

print("ALL COLUMNS: ", col_1, col_2, col_3, col_4, col_5)

## Extract all features into a variable x
x = np.concatenate((col_1, col_2, col_3, col_4), axis=1)
print("Features: ", x)

## Extract price column into a variable y
y = col_5
print("Price column: ", y)

## We can improve with this
# X = data[:, :-1]
# y = data[:, -1]

# Stastical Analysis
## All answers are arrays that shows the stastistics of each column

## Mean for each column
mean = np.nanmean(data, axis=0)
print("Mean: ", mean)

## Median
median = np.nanmedian(data, axis=0)
print("Median: ", median)

## Minimum
minimum = np.nanmin(data, axis=0)
print("Minimum: ", minimum)

## Maximum
maximum = np.nanmax(data, axis=0)
print("Maximum: ", maximum)

## Standard deviation
std = np.nanstd(data, axis=0)
print("Standard deviation: ", std)

# Handle missing values

## Indexes of missing values in the dataset
print("Indexes that have nan value: ", np.argwhere(np.isnan(data)))

## Count the missing values ineach column
missing_values = np.isnan(data).sum(axis=0)
print("Number of missing values per column: ", missing_values)

## Median of each features with ignoring missing values
nan_median = np.nanmedian(data, axis=0)
print("The median of each column with ignoring the nan values: ", nan_median)

## Replace missing feature values with their respective column medians
replaced_data = np.where(np.isnan(data), nan_median, data)
print("Replaced Data: ", replaced_data)

## verify no missing feature values remain
if not np.isnan(replaced_data).sum():
    print("Verified!, There is no missing value.")
else: print("Not verified!, There is a missing value.")

## We can improve with this
# if np.isnan(replaced_data).any():
#     print("Missing values still exist.")
# else:
#     print("Verified! No missing values.")
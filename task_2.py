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

# Boolean filtering

## Houses with an area greater than 100m^2
big_houses = data[data[:, 0] > 100]
print("\n\n Houses with 100 and greater area: \n", big_houses)

## Houses with at least 3 bedrooms
has_bed_houses = data[data[:, 1] >= 3]
print("\n\n Houses with at least 3 bedrooms: \n", has_bed_houses)

## Houses with prices above 250
more_price = data[data[:, -1] > 250]
print("\n\n House with more price: \n", more_price)

## Houses with an area greater than 100 and prices above 250.
more_price_area = data[((data[:, 0] > 100) & (data[:, -1] > 250))]
print("\n\n House with more price and area: \n", more_price_area)

## Houses with an area between 70 and 120 m^2
between_area = data[((data[:, 0] > 70) & (data[:, 0] < 120))]
print("\n\n House with area between 70 and 120: \n", between_area)

# Feature Engineering

## Price price per square meter
area = replaced_data[:, 0]
price = replaced_data[:, -1]
price_per_square = price / area

## Bedrooms per 100 m^2
bedrooms = replaced_data[:, 1]
bedrooms_per_100m2 = (bedrooms / area) * 100

## Added those columns into data
enriched_data = np.column_stack((replaced_data, price_per_square, bedrooms_per_100m2))
print("\n\nAdded column to the data: \n", enriched_data)

## Creating new features is essential for machine learning to improve predictions by giving additial information about each attribute.

# Standardization   
## standardize the area feature
mean_area = np.mean(area)
std_area = np.std(area)
standardized_area = (area - mean_area) / std_area
print(standardized_area)

## standardize the bedroom feature
mean_bedroom = np.mean(bedrooms)
std_bedroom = np.std(bedrooms)
standardized_bedroom = (bedrooms - mean_bedroom) / std_bedroom
print(standardized_bedroom)

## standardize the age feature
age = replaced_data[:, 2]
mean_age = np.mean(age)
std_age = np.std(age)
standardized_age = (age - mean_age) / std_age
print(standardized_age)

## standardized the distance feature
distance = replaced_data[:, 3]
mean_distance = np.mean(distance)
std_distance = np.std(distance)
standardized_distance = (distance - mean_distance) / std_distance
print(standardized_distance)

## merge all the standardized features into one array
standardized_data = np.column_stack((standardized_area, standardized_bedroom, standardized_age,standardized_distance))
standardized_mean = np.mean(standardized_data, axis=0)
standardized_std = np.std(standardized_data, axis=0)
print("\n\nStandardized Data: \n", standardized_data, "\n\nStandardized mean: ", standardized_mean, "\n\nStandardized standard deviation: ", standardized_std)

# Train/Test split
X = replaced_data[:, :-1]
Y = replaced_data[:, -1]
print("\nAll features: \n", X, "\nPrice feature: \n", Y)
rng = np.random.default_rng(seed=43)

indices = rng.permutation(len(X))
split_indices = int(len(X) * 0.8)

train_split = indices[:split_indices]
test_split = indices[split_indices:]

x_train, x_test = X[train_split], X[test_split]
y_train, y_test = Y[train_split], Y[test_split]

print("\nTrain Data Features: \n", x_train, "\nTest Data Features: \n", x_test)
print("\nTrain Data Targets: \n", y_train, "\nTest Data Target: \n", y_test)

# Correlation Analysis
correlation_matrix = np.corrcoef(replaced_data, rowvar=False)
print('\nCorrelation Matrix: \n', correlation_matrix)

## From this we can know that area and price have strong  positive correlation.
## And also we can know that distance and price have strong negative correlation.

## Correlation close to 1 means they have strong positive relation means that they are directly related.
## Correlation close to -1 means they have strong negative relation means that they are inversly related.
## A correlation close to zero means there is little or no linear relationship. However, the variables could still have a nonlinear relationship.

# Baseline prediction
y_mean = np.mean(y_train)
mae = np.mean(np.abs(y_test - y_mean))
print(mae)
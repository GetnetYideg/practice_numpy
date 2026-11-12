import numpy as np

data = np.array([
    [2, 60, 45],
    [4, 70, 55],
    [5, 80, 65],
    [6, 85, 72],
    [8, 90, 85],
    [10, 95, 95]
])

#Dataset underestandingf

## shape
print("Shape: ", data.shape)

## Number of student
print("Number of student: ", len(data))

## Extract the first column
col_1 = data[:, 0:1]
print("Study hours: ", col_1)

## Extrract the second column
col_2 = data[:, 1:2]
print("Attendance: ", col_2)

## Extract the third column 
col_3 = data[:, 2:3]
print("Scores: ", col_3)

# Statistical analysis

## Avarage of Exam score
avarage = np.mean(data, axis=0)
score_avarage = avarage[-1]
print("Avarage score: ", score_avarage)

## Lowest exam score
min_score = np.min(data, axis=0)[-1]
print("Minimum score: ", min_score)

## Highest exam score
max_score = np.max(data, axis=0)[-1]
print("Maximum score: ", max_score)

## Avarage study hours 
avarage_hours = avarage[0]
print("Avarage study hours: ", avarage_hours)

## Avarage attendance
avarage_attendance = avarage[1]
print("Avarage attendance: ", avarage_attendance)

# Data filtering

## Scores greater than 70
high_scores = data[data[:, 2] > 70]
print("Students has a score greater than 70: \n", high_scores)

## Studied at least 6 hours
studied_more = data[data[:, 0] >= 6]
print("Students studied at least 6 hours: \n", studied_more)

## Attended at least 80%
attended_more = data[data[:, 1] >= 80]
print("Students studied at least 80% of class: \n", attended_more)

## Scored above 70 and attended at least 80%
both_criteria = data[((data[:, 2] > 70) & (data[:, 1] >= 85))]
print("Students who scored above 70 and attended at least 85%: \n", both_criteria)

# Feature Enginerring
## Create a new column 

new_column = []
for i in range(len(data)):
    efficiency = data[i][2] / data[i][0]
    new_column.append(efficiency)
data_with_efficiency = np.column_stack((data, new_column))
print("The modified data: \n", data_with_efficiency)

## correction

# efficiency = data[:, 2] / data[:, 0]

# data_with_efficiency = np.column_stack(
#     (data, efficiency)
# )

# Standardization
study_std = np.std(data, axis=0)[0]
study_mean = np.mean(data, axis=0)[0]
standard_study_hour = []
for i in range(len(data)):
    standard_hour = float((data[i][0] - study_mean) / study_std)
    standard_study_hour.append(standard_hour)
print("Standard study hour for each student: ", standard_study_hour)

## Correction

# study_hours = data[:, 0]

# study_mean = np.mean(study_hours)
# study_std = np.std(study_hours)

# standardized_hours = (
#     study_hours - study_mean
# ) / study_std

# print(standardized_hours)
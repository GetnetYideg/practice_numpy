import numpy as np
a = np.array([[[1],[ 2]], [[3], [4]], [[5],[9]]])
print (a.ndim)
print(a.shape)
b = a.reshape(1, 6)
print(b.shape)
print(b)
print(a.flatten())
print(a)
print(a.ravel())
print(a) 
print("==========--+--=========")
marks = np.array([
    [70, 80, 90],
    [60, 75, 85],
    [90, 95, 100]
])
print(marks.shape)
print(marks.transpose())
print(marks.ravel())
print("=======\n||\n||\n||\n=======")
print(np.sum(marks, axis=0))
print(np.min(marks))
print(np.max(marks))
print(np.mean(marks))
print(np.var(marks))
print(np.std(marks))
print(marks.sum(axis=0))
print(marks[(marks > 80) & (marks < 100)])
print(marks[~(marks > 80)])
print(np.sum(marks < 80))

q = np.array([10, 20, 30])
r = np.array([40, 50, 60])

result = np.concatenate((q, r))
print(np.vstack((q, r)))
print(np.hstack((q, r)))

arege = np.split(result, 3)
print(arege)

r1 = r.copy()
r2 = r.view()
print('origina', r)
r1[1] = 12
print("copy", r)
r2[1] = 18
print("view", r)
#slicing always creates a view
import numpy as np

a = np.array([[1, 2], [2, 3]])
b = np.array([[1, 2, 3], [2, 3, 1]])

print(a@b)
print(np.dot(a, b))
print(a * a)
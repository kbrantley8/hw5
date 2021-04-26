import numpy as np

b = 0.21
w = [-0.32, 0.32, 0.11, 0.01]
x = [2, 7, 0, np.log(67)]

weighted_sum = np.sum(np.multiply(w, x)) + b
sigmoid = (1 / (1 + np.exp(-1 * weighted_sum)))
print(sigmoid)

# y = 1
# a = 0.88
# w = [2,7,0,np.log(67)]

# first = np.full(len(w), a)
# first = first - y
# res = np.multiply(w, first)
# print(res)
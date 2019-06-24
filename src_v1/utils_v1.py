import numpy as np

'''
The Goodall3 measure between 2 two categorical variables:
if x == y : 1 - p(x)^2
if x != y : 0  
Todo : memoize
Input : 
data_1 [ np.array ]
data_2 [ np.array ]
prob_list [ np.array ]    # p(x)
weight_list [ np.array ]  # weight of the dimensions
'''
def Goodall3(
        data_1,
        data_2,
        prob_list,
        weight_list
):
    p_w = 1 - np.power(prob_list, 2)
    xy = np.logical_not(data_1 - data_2) + 0
    xy = np.multiply(xy, p_w)
    xy = np.sum(np.multiply(xy, weight_list))
    return xy
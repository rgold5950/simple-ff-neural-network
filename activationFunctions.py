#coding up actiavtion funcitons and their derivatives

import numpy as np

#activation function and its derivatives
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2
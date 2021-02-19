# The Newton Raphson Method is an algorithm to find the root of a function,
# which produces successively better approximations to the roots of a real
# valued function.
# This method is one of many used to find the roots of a function

from sympy import *
import math
import numpy as np

# Newton Raphson Method
def newton_raphson(x_init,f,accuracy):
    # input: inital starting guess for the root
    #        function used in symbolic view to calculate derivative
    #        accuracy for break point

    # calculate the derivative
    f_prime = f.diff(x)

    # create lambda functions
    f = lambdify(x, f)
    f_prime = lambdify(x, f_prime)

    # variable initialization
    # Note that if the root of the function is at 0, need to change the
    # initialization of the x_before variable to some other integer
    x_res = x_init
    x_before = 0 # calculation of previous root
    iterations = 0

    while np.abs(x_res - x_before) > accuracy:
        x_before = x_res
        x_res = x_res - f(x_res)/f_prime(x_res)
        iterations += 1

    return [iterations,x_res]

# Example for the function: f(x) = x - 3^(-x)

x = Symbol('x')
f = x - 3 **(-x)

iterations, root = newton_raphson(0.5,f,10**-10)
print('Number of iterations:',iterations)
print('Root:',root)

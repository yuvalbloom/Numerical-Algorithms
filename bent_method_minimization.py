# The parabolic search method (Bent's Method) is an algorithm to find the minima of a 
# given function. There are many other algorithms, such as the Golden Section Method.
# These algorithms are used to find the minima of 1D funcions and are not useful for multiple dimensions

# To view the example, run the code.

#imports

import numpy as np
import scipy
from scipy.optimize import fmin
import math

# this function uses the parabolic search method to find the minimum of a given function
def parab_min(f, rng, prec):
    # input: f is the function of interest, rng is the range of the function
    #        prec is the precision for the minimum to be used for the breakpoint
    # returns the minimum of the function and its value for the given precision

    # variable initialization
    x_a = rng[0]
    x_b = rng[1]
    x_c = (x_a + x_b)/2 # middle point for first iteration
    x_min = x_c
    x_before = 0 # x before initialized for the breakpoint of the loop
    
    while (abs(x_min - x_before) > prec):
        # function values at range at every iteration
        f_a = f(x_a)
        f_c = f(x_c)
        f_b = f(x_b)
        # last minimum value found for the break point condition
        x_before = x_min
        # finding the minimum using the lagrange method
        x_min = x_c - (1/2)*((((x_c - x_a)**2)*(f_c - f_b) - ((x_c - x_b)**2)*(f_c - f_a))/(((x_c - x_a))*(f_c - f_b) - ((x_c - x_b))*(f_c - f_a)))
        # value of func at the minimum calculated using the lagrange method
        f_min = f(x_min)
        
        # conditions to converge to the minimum point using the parabolic search method
        # if x_min which was calculated is in [c,b]
        if (x_min > x_c):
            if (f_min < f_b and f_min < f_c):
                x_a, x_c = x_c, x_min
            elif (f_min < f_b and f_min > f_c):
                x_b = x_min
                
       # if x_min which was calculated is in [a,c] 
        elif (x_min < x_c):
            if (f_min < f_a and f_min < f_c):
                x_b, x_c = x_c, x_min
            elif (f_min < f_a and f_min > f_c):
                x_a = x_min

    return [x_min,f(x_min)]

##############################################################################################################################
# Example - run this code to print results
f = lambda x: -np.sin(4*(x**2)) # for range [0.1,1]
prec_1 = 1e-2
prec_2 = 1e-5
# analytical minimum solving for the derivative of f
real_min = (np.sqrt(np.pi/2))/2
# calculations using the function
[x_min1, f_min1] = parab_min(f, [0.1,1], prec_1)
[x_min2, f_min2] = parab_min(f, [0.1,1], prec_2)

print('(1) Precision of: ', prec_1,'\n','x minimum: ', x_min1, '\n', 'f(x_min): ', f_min1)
print()
print('(2) Precision of: ', prec_2,'\n','x minimum: ', x_min2, '\n', 'f(x_min): ', f_min2)
print()
print('Analytical Minimum of the function is: ','\n','x minimum: ', real_min, '\n', 'f(x_min): ', f(real_min))
    
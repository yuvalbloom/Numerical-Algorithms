# Computational Integration Method using the Recursive Trapezoid Integration Algorithm

# The recursive trapezoid method is defined as:
#  R(N + 1) = 1/2 * R(N) + h(n+1) * sum(f(a + (2i - 1)*h(n+1)))
# Where h is the height of the trapezoid, a is the lower bound of the integral's range,
# f is the function in the integral, and i is the running index in the summation

#imports
import numpy as np
from numpy import trapz, sqrt

# function which calculates the N'th step of the recursive trapezoid method
def trap_n(N, xi, xf, f):
    # input: N is the number of points used, xi and xf are the boundaries of the function
    #        f is the function of interest
    h = (xf - xi) / (2**(N-1))   
    sum_part = 0  # sum initialized as zero
    for i in range(1, 2**(N-2) + 1):
        sum_part += f(xi + (2*i-1)*h)
    return h * sum_part

def trap_calc(I_exact, xi, xf, f, err):
    # input: I_exact is the analytical solution of the integral, xi and xf are the boundaries of the function
    #        f is the function of interest
    #        err is the precision given for the breakpoint
    N = 2 # beginning step
    I = ((xf - xi)/2) * (f(xi) + f(xf)) # initial step calculation as shown in the recursive function for R(0)
    while (np.abs(I_exact - I) >= err):
        I = (0.5) * I  + trap_n(N, xi, xf, f) # recursive calculation using the recursive formula for each step 
        N += 1
    return I, N - 1
    
######################################################################################################################
# Example
f = lambda x : 1 / (6 + 4 * np.cos(2*x))
xi = 0
xf = np.pi / 2
err = 10 **(-16) # double error precision

n = 2**(6) + 1
x = np.linspace(0, np.pi/2, n)
y = f(x)

I_trapz = trapz(y, x) # python's trapz method

g = lambda x : np.arctan(np.tan(x)/sqrt(5)) / (2*sqrt(5)) # Analytical calculation of the integral of f
I_ana = g(np.pi / 2) - g(0) # Analytical Integral Calculation

I_rectrap, steps = trap_calc(I_ana, xi, xf, f, err)

# Recursive Trapezoid Method Integral Solution
print('The Recursive Trapezoid Method Integral Solution is: ', "%.20f" %I_rectrap)

# Number of steps to Calculate the Integral
print('Number of steps: ', steps)

# Numpy.trapz Integral Solution
print('The Numpy.Trapz Method Integral Solution is: ', "%.20f" %I_trapz)

# Analytical Integral Solution
print('The Analytical Integral Solution is: ', "%.20f" %I_ana)

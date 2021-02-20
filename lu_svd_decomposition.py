# LU decomposition algorithm to solve linear equations:
# This algorithm finds the upper (U) and lower (L) matrices as defined: A = L*U,
# where A is a N x N matrix. Using this algorithm, one can solve linear equations as defined Ax = b

# SVD decomposition algorithm can solve linear equations for singular matrices or a matrix which is M x N
# In this script, I compare between this two methods for solving linear equations
# To see the results from the examples, run the code. 

#imports
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.linalg import lu
from numpy import sqrt, sin, cos, pi, abs, arctan,tan, linspace, polyfit
from numpy import linalg

def lu(A):
    # input: a squared matrix
    # returns the L & U matrices as described
    size = np.shape(A) # size of matrix A
    N = len(A) # length of the matrix N x N
    # Initialization of the L & U matrixes as N x N zeros
    L = np.zeros(size)
    U = np.zeros(size)
    # loop through the matrixes to calculate the LU decomposition using Crout's algorithm
    # loop through each column
    for j in range(N):
        # loop through each row
        for i in range(j+1):
            # calculate U
            U[i,j] = A[i,j] - sum(U[k,j] * L[i,k] for k in range(i))
        for i in range(j,N):
            # calculate L
            L[i,j] = 1/(U[j,j]) * (A[i,j] - sum([U[k,j]*L[i,k] for k in range(j)]))
            
    return [L,U]

# This function calculates the inverse matrix and the x vector for linear equations,
# Using the LU or SVD methods for a given threshold
# The SVD method used here is executed from the numpy library

def LUorSVD(A,b,thres = 10**5):
    # input: A is a sqaured matrix, b is the solutions vector
    #        thres is the threshold to decide whether the matrix is singular
     
    # check if the matrix is singular
    svals = scipy.linalg.svdvals(A) # find all the singular points of the matrix
    ratio = max(svals) / min(svals) # find the ratio of the maximum and minimum value of the singular points
    
    # if not singular, use the LU decomposition method
    if (ratio < thres):
        l, u = lu(A)
        method = 'LU'
        
        y = np.zeros(len(A)) # initialize a vector of zeros for y
        # loop for the calculation for the matrix length, using forward substitution
        for i in range(len(A)):
            y[i] = (1/l[i,i])*(b[i] - sum(y[j] * l[i,j] for j in range(i)))
        
        x = np.zeros(len(A)) # initialize a vector of zeros for x
        # loop for the calculation for the matrix length, using backward substitution
        for i in range(len(A)-1,-1,-1):
            x[i] = (1/u[i,i])*(y[i] - sum(x[j] * u[i,j] for j in range(i+1,len(A))))
        
        # find the inverse of A
        l_inv = np.linalg.inv(l)
        u_inv = np.linalg.inv(u)
        
        Ainv = np.dot(u_inv,l_inv)
        
    # if singular, use the SVD decomposition method
    elif (ratio >= thres):
        # calculate the SVD using the numpy library
        [U,S,V] = np.linalg.svd(A)
        method = 'SVD'
        
        Snew = (S>10**-5).astype(int)*S 
        Sinv=np.concatenate((1/Snew[Snew>0],Snew[Snew<=0])) # calculate the inverse of S
        n=len(Sinv)
        
        Ainv = V.T.dot((Sinv*np.eye(n)).dot(U.T)) # calculate A inverse using the matrixes calculated in the SVD method
        x = Ainv.dot(b) # calculate x using the Ainv and b
        
    return [x, Ainv, method]
    
########################################################################################################################
# Example - Run this code to print the result
# Matrixes and solution vector initializtion
# 1
A1 = np.array([[1,2,3],[4,5.5,6],[7,8,9]])
b1 = np.array([[1],[3],[5]])

# 2
A2 = np.array([[1,2,3],[2,4,6],[7,8,9]])
b2 = np.array([[1],[3],[5]])

# 3
A3 = np.array([[1,2,3],[4,5,6],[5,7,9]])
b3 = np.array([[1],[3],[5]])

example_1 = LUorSVD(A1,b1)
example_2 = LUorSVD(A2,b2)
example_3 = LUorSVD(A3,b3)

# 1
print('First Result')
print('Method used: ',example_1[2],'\n', 'x vector:','\n',example_1[0],'\n','Ainv:','\n', example_1[1],'\n','b vector:','\n',np.dot(A1,example_1[0]))
print()
# 2
print('Second Result')
print('Method used: ',example_2[2],'\n', 'x vector:','\n',example_2[0],'\n','Ainv:','\n', example_2[1],'\n','b vector:','\n',np.dot(A2,example_2[0]))
print()
# 3
print('Third Result')
print('Method used: ',example_3[2],'\n', 'x vector:','\n',example_3[0],'\n','Ainv:','\n', example_3[1],'\n','b vector:','\n',np.dot(A3,example_2[0]))
    

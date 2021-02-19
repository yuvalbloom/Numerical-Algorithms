# Gauss Jordan Elimination
# Algorithm to solve linear equations of the kind A * x = b, where A is the matrix, and b is solutions vector
# This algorithm was computed using partial pivoting - can be created without pivoting (usually not customary for these
# calculations) by deleting the block of code used to swap the rows.

#imports
import numpy as np
import math
import sympy
import pandas as pd

def gauss_jordan(A,b):
    # input: A is a matrix N x N, and b is a vector the length of N
    
    # Matrix Initialization
    I = np.eye(len(A)) # Identity Matrix the size of A
    A_tot = np.concatenate((A, b, I), axis=1) # Matrix to perform the Gauss Jordan elimination on
    
    k = 0 # index for the loop
    
    # loop through all the rows of the matrix
    for i in range(len(A_tot)):
        
        # PIVOT SECTION
        # pivot for every column
        piv = abs(A_tot[i,k])
        index = i # temp index for swapping

        # loop through the column to find the pivot
        for n in range(k,len(A_tot)):
            if (abs(A_tot[n,k]) > piv):
                piv = abs(A_tot[n,k])
                index = n
        # swapping the rows
        A_tot[[i, index]] = A_tot[[index, i]]
            
        # normalize the row which is used in this iteration
        A_tot[i] = (1/A_tot[i,k])*A_tot[i]
        
        # GAUSS JORDAN SECTION
        # loop through all the rows where i != j for the Gauss Jordan elimination for every row
        for j in range(len(A_tot)):
            if (i != j):
                # calculation for every row
                A_tot[j] = A_tot[j] - (A_tot[i]*A_tot[j,k])
        
        k += 1 # index progress
        
    # Slicing the matrix to return the inverse matrix and x vector separtely
    Ainv = A_tot[:,k+1:]
    x = A_tot[:,k]  
    
    return [Ainv, x]
    
# Example
# RUN THIS CODE TO SEE EXAMPLE
# initializing an example matrix
mat = "12.113 1.067 9.574 8.414 0.098 -0.046; 9.609  5.015 8.814 7.983 7.692 -11.655; 7.402  0.081 5.394 0.417 9.603 0.0623; 1.451  1.517 3.741 4.668 2.601 -1.351; 2.053  1.576 8.046 8.152 2.896 -0.227"
m_np = np.matrix(mat) # create the matrix

solve_gj = gauss_jordan(m_np[:,:5],m_np[:,5])

b_sol = m_np[:,:5] * solve_gj[1]
x_sol = solve_gj[0] * m_np[:,5]
identity_mat_sol = m_np[:,:5] * solve_gj[0] 

# Show Results for the example
print('Gauss Jordan Elimination with partial pivoting:')
print('Ainv: ')
print(pd.DataFrame(solve_gj[0]))
print('x: ')
print(pd.DataFrame(solve_gj[1]))
print('A*x: ')
print(pd.DataFrame(b_sol))
print('Ainv*b: ')
print(pd.DataFrame(x_sol))
print('A*Ainv: ')
print(pd.DataFrame(identity_mat_sol))




    
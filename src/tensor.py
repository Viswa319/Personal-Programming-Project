# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
class tensor():
    
    
    def __init__(self):
        pass
    def P4sym(self):
        I = np.eye(3)
        I4sym = np.zeros((3,3,3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        I4sym[i,j,k,l] = (1/2) * ((I[i,k]*I[j,l]) + (I[i,l]*I[j,k]))
        
        P4sym = np.zeros((3,3,3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        P4sym[i,j,k,l] = I4sym[i,j,k,l] - (1/3)*(I[i,j]*I[k,l])
                        
        return P4sym
    
    def t2_otimes_t2(self,A,B):
        n = len(A)
        C4 = np.zeros(n,n,n,n)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        C4[i,j,k,l] = A[i,j]*B[k,l]
        
        return C4
    
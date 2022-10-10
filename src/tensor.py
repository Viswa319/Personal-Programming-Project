# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
class tensor():
    """This class consists of some tensor operations. 
    
    This part of code is directly taken from **Prof. Kiefer**, provided in the plasticity exercise.
    """
    
    def __init__(self):
        pass
    def P4sym(self):
        """
        Function to compute fourth order deviatoric projection tensor.

        Returns
        -------
        P4sym : Array of float64, size(3,3,3,3)
            Fourth order deviatoric projection tensor.

        """
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
    
    def fourth_to_three(self,C:np.array):
        """
        Function to reshape fourth order tensor from 3 x 3 x 3 x 3 to 6 x 6.
        Then 6 x 6 order tensor is reduced to 3 x 3 order tensor based on plane strain conditions.

        Parameters
        ----------
        C : Array of float64, size(3,3,3,3)
            Fourth order tensor.

        Returns
        -------
        C_red : Array of float64, size(3,3)
            Second order tensor.

        """
        ii = [0,1,2,0,1,0]
        jj = [0,1,2,1,2,2]
        A66 = np.zeros((6,6))
        for i in range(6):
            for j in range(6):
                A66[i,j] = C[ii[i],jj[i],ii[j],jj[j]]
        
        # Reduce stiffness tensor into a 3 x 3 matrix
        C_red = np.copy(A66)
        C_red = np.delete(C_red,[2,4,5],axis = 0)
        C_red = np.delete(C_red,[2,4,5],axis = 1)
        return C_red
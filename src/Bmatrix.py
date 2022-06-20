# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
class Bmatrix:
    def __init__(self):
        pass
    def Bmatrix_linear(self,dNdX,num_node_elem):
        """
        Function for computing strain and displacement connectivity matrix (B)

        Parameters
        ----------
        dNdX : Array of float64, size(num_dim,num_node_elem)
            derivatives of shape functions w.r.t. global coordinates.
        num_node_elem : int
            number of nodes per element, possible nodes are 2,3,4 and 8 per element.
        
        Returns
        -------
        B : Array of float64, size(3,2*num_node_elem)
            strain and displacement connectivity matrix of a node.

        """
        B = np.zeros((3, 2*num_node_elem))
        for I in range(0, num_node_elem):
            J = 2*I
            K = 2*I+1
            B[0, J] = B[2, K] = dNdX[0, I]# dNdx[i]
            B[0, K] = B[1, J] = 0
            B[1, K] = B[2, J] = dNdX[1, I]# dNdy[i]
        return B
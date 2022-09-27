# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
class Bmatrix:
    def __init__(self,dNdX,num_node_elem):
        self.dNdX = dNdX
        self.num_node_elem = num_node_elem
    def Bmatrix_disp(self):
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
        B = np.zeros((3, 2*self.num_node_elem))
        for i in range(0, self.num_node_elem):
            j = 2*i
            k = 2*i+1
            B[0, j] = B[2, k] = self.dNdX[0, i]# dNdx[i]
            B[0, k] = B[1, j] = 0
            B[1, k] = B[2, j] = self.dNdX[1, i]# dNdy[i]
        return B
    
    def Bmatrix_phase_field(self):
        B = self.dNdX
        return B
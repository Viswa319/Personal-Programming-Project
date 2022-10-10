# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
class shape_function:
    """Class for computing shape function, derivatives of shape function and determinant of Jacobian.
    
    Shape functions are computed using Lagrange interpolant basis.
    
    Derivatives of shape functions are computed using gradients of Lagrange interpolant basis.
    
    Lagrange interpolant basis are implemented for four different cases. 
    
    1. Two node line element.\n
    2. Three node triangular element.\n
    3. Four node quadrilateral element.\n
    4. Eight node quadrilateral element.
    """
    def __init__(self,num_node_elem:int,gpos:list,elem_coord:np.array):
        self.num_node_elem = num_node_elem
        self.gpos = gpos
        self.elem_coord = elem_coord
        if num_node_elem == 2:
            ########### two node line element ########## 
            #  
            #    1---------2
            #
            # Shape functions
            self.N = np.array([(1+self.gpos)/2,(1-self.gpos)/2])[np.newaxis] 
            
            # derivatives of shape functions w.r.t psi
            self.dNdxi = np.array([1/2,-1/2])[np.newaxis]

        elif num_node_elem == 3:
            ########### three node triangular element ########## 
            #  
            #    1---------2
            #     \       /
            #      \     /
            #       \   /
            #        \ /
            #         3
            #
            # Shape functions
            psi = self.gpos[0]
            eta = self.gpos[1]
            self.N = np.array([1-psi-eta,psi,eta])[np.newaxis] 
            
            # derivatives of shape functions w.r.t psi and eta respectively
            dNdpsi = np.array([-1, 1, 0])[np.newaxis]
            dNdeta = np.array([-1, 0, 1])[np.newaxis]
            self.dNdxi = np.r_[dNdpsi,dNdeta]

        elif num_node_elem == 4:
            ########### four node quadrilateral element ########## 
            #  
            #    4---------3
            #    |         | 
            #    |         |
            #    |         |
            #    1---------2
            # Shape functions
            psi = self.gpos[0]
            eta = self.gpos[1]
            self.N = 0.25*np.array([(1-psi)*(1-eta), (1+psi)*(1-eta), (1+psi)*(1+eta), (1-psi)*(1+eta)])[np.newaxis]
        
            # derivatives of shape functions w.r.t psi and eta respectively
            dNdpsi = 0.25*np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])[np.newaxis]
            dNdeta = 0.25*np.array([-(1-psi), -(1+psi), (1+psi), (1-psi)])[np.newaxis]
            self.dNdxi = np.r_[dNdpsi,dNdeta]

        elif num_node_elem == 8:
            ########### eight node quadrilateral element ########## 
            #
            #    7-----6-----5
            #    |           | 
            #    |           |
            #    8           4 
            #    |           |
            #    |           |
            #    1-----2-----3
            # Shape functions
            psi = self.gpos[0]        
            eta = self.gpos[1]
            self.N = np.array([-1*0.25*(1-psi)*(1-eta)*(1+psi+eta), 0.5*(1-psi)*(1-eta)*(1+psi), -1*0.25*(1+psi)*(1-eta)*(1-psi+eta), 0.5*(1+psi)*(1-eta)*(1+eta), -1*0.25*(1+psi)*(1+eta)*(1-psi-eta), 0.5*(1+psi)*(1+eta)*(1-psi),-1*0.25*(1-psi)*(1+eta)*(1+psi-eta), 0.5*(1-psi)*(1+eta)*(1-eta)])[np.newaxis]
            
            # derivatives of shape functions w.r.t psi and eta respectively
            dNdpsi = np.array([0.25*(1-eta)*(2*psi+eta), -psi*(1-eta), 0.25*(1-eta)*(2*psi-eta), 0.5*(1-eta)*(1+eta), 0.25*(1+eta)*(2*psi+eta),-psi*(1+eta),0.25*(1+eta)*(2*psi-eta),-0.5*(1-eta)*(1+eta)])[np.newaxis]
            dNdeta = np.array([0.25*(1-psi)*(psi+2*eta), -0.5*(1-psi)*(1+psi), 0.25*(1+psi)*(-psi+2*eta), -eta*(1+psi), 0.25*(1+psi)*(psi+2*eta),0.5*(1-psi)*(1+psi),0.25*(1-psi)*(-psi+2*eta),-eta*(1-psi)])[np.newaxis]
            self.dNdxi = np.r_[dNdpsi,dNdeta]
            
        self.Jacobian = np.matmul(self.dNdxi,self.elem_coord)
            
    def get_shape_function(self):
        """
        Function to compute shape functions using Lagrange interpolant basis.

        Returns
        -------
        Array of float64
            Shape functions.

        """
        return self.N

    def get_det_Jacobian(self):
        """
        Function to compute determinant of Jacobian matrix.

        Raises
        ------
        ValueError
            Determinant of Jacobian matrix should be positive.

        Returns
        -------
        float64
            Determinant of Jacobian matrix.

        """
        self.det_Jacobian = np.linalg.det(self.Jacobian)
        if self.det_Jacobian <= 0:
            raise ValueError('Solution is terminated since, determinant of Jacobian is either zero or negative.')
        return self.det_Jacobian
    
    def get_shape_function_derivative(self):
        """
        Function to compute derivatives of shape functions using inverse of 
        Jacobian matrix and gradients of Lagrange interpolant basis.

        Returns
        -------
        Array of float64
            Derivatives of shape functions.

        """
        Jacobian_inv = np.linalg.inv(self.Jacobian)
        self.dNdX = np.matmul(Jacobian_inv,self.dNdxi)
        return self.dNdX
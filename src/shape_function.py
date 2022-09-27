# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
class shape_function:
    def __init__(self,num_node_elem,gpos,elem_coord):
        self.num_node_elem = num_node_elem
        self.gpos = gpos
        self.elem_coord = elem_coord
        if num_node_elem == 2:
            """
            Function to get the Lagrange interpolant basis and its gradients w.r.t its coordinates 
            for a two node line element.
            
    
            Parameters
            ----------
            point : array of float64
            point.
    
            Returns
            -------
            N : array of float64
                Lagrange interpolant shape function.
            dNdxi : array of float64
                gradient of Lagrange interpolant shape function w.r.t respecive coordinates.
    
            """
            ########### two node line element ########## 
            #  
            #    1---------2
            #
            # Shape functions
            self.N = np.array([(1+self.gpos)/2,(1-self.gpos)/2])[np.newaxis] 
            
            # derivatives of shape functions w.r.t psi
            self.dNdxi = np.array([1/2,-1/2])[np.newaxis]

        elif num_node_elem == 3:
            """
            Function to get the Lagrange interpolant basis and its gradients w.r.t its coordinates 
            for a three node triangular element.
    
            Parameters
            ----------
            point : array of float64
                point.
    
            Returns
            -------
            N : array of float64
                Lagrange interpolant shape function.
            dNdxi : array of float64
                gradient of Lagrange interpolant shape function w.r.t respecive coordinates.
    
            """
    
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
            """
            Function to get the Lagrange interpolant basis and its gradients w.r.t its coordinates 
            for a four node quadrilateral element.
    
            Parameters
            ----------
            point : array of float64
                point.
    
            Returns
            -------
            N : array of float64
                Lagrange interpolant shape function.
            dNdxi : array of float64
                gradient of Lagrange interpolant shape function w.r.t respecive coordinates.
    
            """
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
            """
            Function to get the Lagrange interpolant basis and its gradients w.r.t its coordinates 
            for an eight node quadrilateral element.
    
            Parameters
            ----------
            point : array of float64
                point.
    
            Returns
            -------
            N : array of float64
                Lagrange interpolant shape function.
            dNdxi : array of float64
                gradient of Lagrange interpolant shape function w.r.t respecive coordinates.
    
            """
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
        return self.N

    def get_det_Jacobian(self):
        
        self.det_Jacobian = np.linalg.det(self.Jacobian)
        if self.det_Jacobian <= 0:
            raise ValueError('Solution is terminated since, determinant of Jacobian is either zero or negative.')
        return self.det_Jacobian
    
    def get_shape_function_derivative(self):
        Jacobian_inv = np.linalg.inv(self.Jacobian)
        self.dNdX = np.matmul(Jacobian_inv,self.dNdxi)
        return self.dNdX
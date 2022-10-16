# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
class Lagrange_interpolant:
    def __init__(self):
        pass
    def two_node_line_element(self,point):
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
        N = np.array([(1+point)/2,(1-point)/2])[np.newaxis] 
        
        # derivatives of shape functions w.r.t psi
        dNdxi = np.array([1/2,-1/2])[np.newaxis]
        return N,dNdxi
    
    def three_node_triangular_element(self,point):
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
        psi = point[0]
        eta = point[1]
        N = np.array([1-psi-eta,psi,eta])[np.newaxis] 
        
        # derivatives of shape functions w.r.t psi and eta respectively
        dNdpsi = np.array([-1, 1, 0])[np.newaxis]
        dNdeta = np.array([-1, 0, 1])[np.newaxis]
        dNdxi = np.r_[dNdpsi,dNdeta]
        return N,dNdxi
    def four_node_quadrilateral_element(self,point):
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
        psi = point[0]
        eta = point[1]
        N = 0.25*np.array([(1-psi)*(1-eta), (1+psi)*(1-eta), (1+psi)*(1+eta), (1-psi)*(1+eta)])[np.newaxis]
    
        # derivatives of shape functions w.r.t psi and eta respectively
        dNdpsi = 0.25*np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])[np.newaxis]
        dNdeta = 0.25*np.array([-(1-psi), -(1+psi), (1+psi), (1-psi)])[np.newaxis]
        dNdxi = np.r_[dNdpsi,dNdeta]
        return N,dNdxi
    def eight_node_quadrilateral_element(self,point):
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
        psi = point[0]        
        eta = point[1]
        N = np.array([-1*0.25*(1-psi)*(1-eta)*(1+psi+eta), 0.5*(1-psi)*(1-eta)*(1+psi), -1*0.25*(1+psi)*(1-eta)*(1-psi+eta), 0.5*(1+psi)*(1-eta)*(1+eta), -1*0.25*(1+psi)*(1+eta)*(1-psi-eta), 0.5*(1+psi)*(1+eta)*(1-psi),-1*0.25*(1-psi)*(1+eta)*(1+psi-eta), 0.5*(1-psi)*(1+eta)*(1-eta)])[np.newaxis]
        
        # derivatives of shape functions w.r.t psi and eta respectively
        dNdpsi = np.array([0.25*(1-eta)*(2*psi+eta), -psi*(1-eta), 0.25*(1-eta)*(2*psi-eta), 0.5*(1-eta)*(1+eta), 0.25*(1+eta)*(2*psi+eta),-psi*(1+eta),0.25*(1+eta)*(2*psi-eta),-0.5*(1-eta)*(1+eta)])[np.newaxis]
        dNdeta = np.array([0.25*(1-psi)*(psi+2*eta), -0.5*(1-psi)*(1+psi), 0.25*(1+psi)*(-psi+2*eta), -eta*(1+psi), 0.25*(1+psi)*(psi+2*eta),0.5*(1-psi)*(1+psi),0.25*(1-psi)*(-psi+2*eta),-eta*(1-psi)])[np.newaxis]
        dNdxi = np.r_[dNdpsi,dNdeta]
        return N,dNdxi
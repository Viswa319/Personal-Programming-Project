# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
def shape_function(point,numnode_elem):
    """
    Function to get the Lagrange interpolant basis and its gradients w.r.t its coordinates.

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
    if numnode_elem == 2:
        ########### two node line element ########## 
        #  
        #    1---------2
        #
        # Shape functions
        N = np.array([(1+point)/2,(1-point)/2])[np.newaxis] 
        
        # derivatives of shape functions w.r.t psi
        dNdxi = np.array([1/2,-1/2])[np.newaxis]
    elif numnode_elem == 3:
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
        eta = point[1]
        psi = point[0]
        N = np.array([1-psi-eta,psi,eta])[np.newaxis] 
        
        # derivatives of shape functions w.r.t psi and eta respectively
        dNdpsi = np.array([-1, 1, 0])[np.newaxis]
        dNdeta = np.array([-1, 0, 1])[np.newaxis]
        dNdxi = np.r_[dNdpsi,dNdeta]
    elif numnode_elem == 4:
        ########### four node quadrilateral element ########## 
        #  
        #    4---------3
        #    |         | 
        #    |         |
        #    |         |
        #    1---------2
        # Shape functions
        eta = point[1]
        psi = point[0]
        N = 0.25*np.array([(1-psi)*(1-eta), (1+psi)*(1-eta), (1+psi)*(1+eta), (1-psi)*(1+eta)])[np.newaxis]
    
        # derivatives of shape functions w.r.t psi and eta respectively
        dNdpsi = 0.25*np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])[np.newaxis]
        dNdeta = 0.25*np.array([-(1-psi), -(1+psi), (1+psi), (1-psi)])[np.newaxis]
        dNdxi = np.r_[dNdpsi,dNdeta]
    elif numnode_elem == 8:
        ########### eight node quadrilateral element ########## 
        #
        #    7----6-----5
        #    |          | 
        #    |          |
        #    8          4 
        #    |          |
        #    |          |
        #    1----2-----3
        # Shape functions
        eta = point[1]
        psi = point[0]
        N = np.array([-1*0.25*(1-psi)*(1-eta)*(1+psi+eta), 0.5*(1-psi)*(1-eta)*(1+psi), -1*0.25*(1+psi)*(1-eta)*(1-psi+eta), 0.5*(1+psi)*(1-eta)*(1+eta), -1*0.25*(1+psi)*(1+eta)*(1-psi-eta), 0.5*(1+psi)*(1+eta)*(1-psi),-1*0.25*(1-psi)*(1+eta)*(1+psi-eta), 0.5*(1-psi)*(1+eta)*(1-eta)])[np.newaxis]
        
        # derivatives of shape functions w.r.t psi and eta respectively
        dNdpsi = np.array([0.25*(1-eta)*(2*psi+eta), -psi*(1-eta), 0.25*(1-eta)*(2*psi-eta), 0.5*(1-eta)*(1+eta), 0.25*(1+eta)*(2*psi+eta),-psi*(1+eta),0.25*(1+eta)*(2*psi-eta),-0.5*(1-eta)*(1+eta)])[np.newaxis]
        dNdeta = np.array([0.25*(1-psi)*(psi+2*eta), -0.5*(1-psi)*(1+psi), 0.25*(1+psi)*(-psi+2*eta), -eta*(1+psi), 0.25*(1+psi)*(psi+2*eta),0.5*(1-psi)*(1+psi),0.25*(1+psi)*(-psi+2*eta),-eta*(1-psi)])[np.newaxis]
        dNdxi = np.r_[dNdpsi,dNdeta]
    else:
        raise ValueError("Number of nodes per element provided is not possible. Possible nodes per element are 2,3,4 and 8")
    return N,dNdxi

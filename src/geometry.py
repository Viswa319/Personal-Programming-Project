# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
def quadrature_coefficients(numgauss:int):
    """
    Function to get Gaussian quadrature points and weights.
    
    Gaussian points and weights for 1-Dimension are called from inbuilt function using **numpy**.
    
    Using these points and weights in 1-Dimension, points and weights are computed in 2-Dimension respectively.
    
    Parameters
    ----------
    numgauss : int
        number of Guass points for background integration.

    Returns
    -------
    Points : Array of float64, size(num_dim,num_Gauss**num_dim)
            Gauss points used for integration.
    Weights : Array of float64, size(num_Gauss**num_dim)
            Weights for Gauss points used for integration.

    """
    import numpy.polynomial.legendre as quad
    
    # Gauss quadrature points and weights are computed using inbuilt funtion in numpy 
    PtsWts = quad.leggauss(numgauss)
    points = PtsWts[0]
    weights = PtsWts[1]
    
    Points = []
    Weights = []
    # Computing Gauss points and weights in 2-dimension
    for i in range(len(points)):
        for j in range(len(points)):
            Points.append([points[i],points[j]])
            Weights.append(weights[i]*weights[j])
    return Points,Weights

import numpy as np

class Lagrange_interpolant:
    """Class to compute Lagrange interpolant basis and its gradients with respect to its coordinates.
    Which have be used as shape functions and derivatives of shape functions.
    """
    
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

import matplotlib.pyplot as plt
def phi_function():
    """
    This function generates a plot for phase-field order parameter varying the length parameter
    
    Returns
    -------
    None.

    """
    x = np.linspace(-5,5,101)
    l_0 = [0.001,0.05,0.1,0.5,1.0]
    phi = {}
    for i in l_0:
        Phi = np.zeros(len(x))
        for j in range(0,len(x)):
            Phi[j] = np.exp(-abs(x[j])/i)
        phi[i] = Phi.tolist()
        
    fig,ax = plt.subplots()
    for i in phi:
        plt.plot(x,phi[i],label = f'$l$ = {i}')
        ax.set_title('Variation of phase-field parameter $\phi$')
        ax.set_xlabel('\phi')
        ax.set_ylabel('x')
        plt.legend()
        plt.savefig('phi_function.png',dpi=600,transparent = True)
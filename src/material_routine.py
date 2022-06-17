# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
class material_routine:
    
    """
    Class where functions calculate stiffness tensor based on stress state,
    either plane stress or plane strain 

    Returns
    -------
    Ce :  Array of float64, size(k)
        Stiffness tensor.

    """
    def __init__(self):
        pass
    def planestress(self,Young,Poisson):  
        """
        Caculates stiffness tensor for plane stress condition.

        Parameters
        ----------
        Young : int
            Youngs modulus.
        Poisson : float
            Poissons ratio.

        Returns
        -------
        Ce :  Array of float64, size(k)
            Stiffness tensor for plane strain case.
        """
        Ce = np.zeros((3, 3))
        #________Compute material stiffness matrix for plane stress
        you = (Young)/(1-Poisson**2)
        Ce[0, 0] = Ce[1, 1] = you*1
        Ce[0, 1] = Ce[1, 0] = you*Poisson
        Ce[2, 2] = you*((1-Poisson)/2)
        return Ce
    def planestrain(self,Young,Poisson):
        """
        Caculates stiffness tensor for plane strain condition.

        Parameters
        ----------
        Young : int
            Youngs modulus.
        Poisson : float
            Poissons ratio.

        Returns
        -------
        Ce :  Array of float64, size(k)
            Stiffness tensor for plane strain case.
        """
        Ce = np.zeros((3, 3))
        #________Compute material stiffness matrix for plane strain
        you = ((Young)*(1-Poisson))/((1-Poisson*2)*(1+Poisson))
        Ce[0,0] = Ce[1,1] = you*1
        Ce[0,1] = Ce[1,0] = you*(Poisson/(1-Poisson))
        Ce[2,2] = you*((1-2*Poisson)/(2*(1-Poisson)))
        return Ce
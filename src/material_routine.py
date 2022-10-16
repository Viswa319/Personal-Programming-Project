# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
from input_abaqus import input_parameters
from tensor import tensor
tensor = tensor()
class material_routine:
    """Depends on the problem type respective variables are returned.
    
    If the problem is **elastic**, stiffness tensor, which is computed either using 
    plane stress or plane strain function, returned.
    
    If the problem is **elastic-plastic**, radial return stress-update algorithm is implemented and
    updated stress, stiffness tensor and internal variables are returned.
    
    Associative von Mises plasticity with linear isotropic hardening model has been implemented. 

    """
    
    
    def __init__(self,problem:str,load_type = None):
        """
        Inputs for material routine are called using input_parameters class based on the problem type.

        Parameters
        ----------
        problem : str
            problem == 'Elastic' or 'Elastic-Plastic'.

        Returns
        -------
        None.

        """
        inputs = input_parameters(load_type,problem)
        if problem == 0:    
            self.k_const, self.G_c, self.l_0, self.Young, self.Poisson, self.stressState = inputs.material_parameters_elastic()
        
        elif problem == 1 or 2:
            self.k_const, self.G_c, self.l_0, self.stressState, self.shear, self.bulk, self.sigma_y, self.hardening = inputs.material_parameters_elastic_plastic()
            self.Young = (9 * self.bulk * self.shear) / ((3 * self.bulk) + self.shear)
            self.Poisson = ((3 * self.bulk) - (2 * self.shear)) / ((6 * self.bulk) + (2 * self.shear))
        elif problem == 'Monolithic':
            from input_parameters import k_const, Young, Poisson, stressState
            self.k_const = k_const
            self.Young = Young
            self.Poisson = Poisson
            self.stressState = stressState
            
    def planestress(self):  
        """
        Calculates stiffness tensor for plane stress condition.

        Returns
        -------
        Ce : Array of float64, size(3,3)
            Stiffness tensor for plane stress case.

        """
        Ce = np.zeros((3, 3))
        #________Compute material stiffness matrix for plane stress condition
        you = (self.Young)/(1-self.Poisson**2)
        Ce[0, 0] = Ce[1, 1] = you*1
        Ce[0, 1] = Ce[1, 0] = you*self.Poisson
        Ce[2, 2] = you*((1-self.Poisson)/2)
        return Ce
    
    def planestrain(self):
        """
        Calculates stiffness tensor for plane strain condition.

        Returns
        -------
        Ce : Array of float64, size(3,3)
            Stiffness tensor for plane strain case.

        """
        Ce = np.zeros((3, 3))
        #________Compute material stiffness matrix for plane strain condition
        you = ((self.Young)*(1-self.Poisson))/((1-self.Poisson*2)*(1+self.Poisson))
        Ce[0,0] = Ce[1,1] = you*1
        Ce[0,1] = Ce[1,0] = you*(self.Poisson/(1-self.Poisson))
        Ce[2,2] = you*((1-2*self.Poisson)/(2*(1-self.Poisson)))
        return Ce
    
    def material_elasticity(self):
        """
        Stiffness tensor computed using plane strain or plane stress function is called and returned.

        Returns
        -------
        Ce : Array of float64, size(3,3)
            Stiffness tensor.

        """
        if self.stressState == 1:  # Plane stress
            Ce = self.planestress()
        elif self.stressState == 2:  # Plane strain
            Ce = self.planestrain()
        
        return Ce
    
    def material_plasticity(self,strain:np.array, strain_plas:np.array, alpha:float):
        """
        The radial return stress-update algorithm for a **von Mises** plasticity model with linear **isotropic** hardening is implemented.
        
        This particular code has been used from my **plasticity course exercise**, where I have implemented it on my own.
        
        Parameters
        ----------
        strain : Array of float64, size(num_stress)
            Total strain  at current step .
        strain_plas : Array of float64, size(num_stress)
            Plastic strain at old step.
        alpha : float64
            Scalar hardening variable at old step.

        Returns
        -------
        stress_red : Array of float64, size(num_stress)
            Updated stress at current step .
        C_red : Array of float64, size(3,3)
            Updated stiffness tensor.
        strain_plas_red : Array of float64, size(num_stress)
            Plastic strain at current step .
        alpha : float64
            Scalar hardening variable at current step.

        """
        tol = 1e-8
        # Call fourth order deviatoric projection tensor from tensor class
        P4sym = tensor.P4sym()
        # Assigning identity tensor
        I = np.eye(3)
        
        # Call elastic stiffness tensor
        if self.stressState == 1:  # Plane stress
            Ce = self.planestress()
        elif self.stressState == 2:  # Plane strain
            Ce = self.planestrain()
        
        # restore strain vector to a strain tensor at current step (t_n+1)
        strain_new = np.array([[strain[0],strain[2]/2,0],[strain[2]/2,strain[1],0],[0,0,0]])
        # restore plastic strain vector to a tensor at old step (t_n)
        strain_plas_n = np.array([[strain_plas[0],strain_plas[2]/2,0],[strain_plas[2]/2,strain_plas[1],0],[0,0,0]])
        # Scalar hardening variable at old step (t_n)
        alpha_n = alpha
        
        # Compute deviatoric strain tensor
        dev_strain = strain_new - (1 / 3) * np.trace(strain_new) * np.eye(3)
        
        # Compute trial stress using elastic strain
        stress_tr = np.matmul(Ce,(strain_new-strain_plas_n))
        you = ((self.Young) * (self.Poisson)) / ((1 + self.Poisson)*(1 - 2 * self.Poisson))
        stress_tr[2,2] = you*((strain_new[0,0] - strain_plas_n[0,0]) + (strain_new[1,1] - strain_plas_n[1,1]))
        
        # Compute deviatoric stress tensor at trial step
        # dev_stress_tr = 2 * self.shear * (dev_strain - strain_plas_n)
        dev_stress_tr = stress_tr - (1/3)* np.trace(stress_tr) * np.eye(3)
        
        # Trial value of the increase in yield stress (scalar beta)
        beta_tr = self.hardening * alpha_n
        
        # Trial value of deviatoric stress
        xi_tr = dev_stress_tr
        
        # Norm of xi_tr
        norm_xi_tr = np.linalg.norm(xi_tr)
        
        # Flow direction from trial state
        n_tr = xi_tr / norm_xi_tr
        
        # Trial value of the yield function
        yield_tr = norm_xi_tr - np.sqrt(2/3)*(self.sigma_y + beta_tr)
        
        # Decide if elastic or elastic-plastic step
        if yield_tr < tol:
            # Elastic step 
            C_red = Ce
            strain_plas = strain_plas_n  # Plastic strain at new step
            alpha = alpha_n  # Scalar hardening variable at new step
            stress = np.matmul(C_red,(strain_new - strain_plas_n))  # Updated stress
        else:
            # Elastic plastic step
            gamma = yield_tr / (2 * self.shear + (2 / 3)*self.hardening)
            dev_stress = dev_stress_tr - (2 * self.shear * gamma * n_tr)  # deviatoric stress
            alpha = alpha_n + gamma * np.sqrt(2 / 3)  # Scalar hardening variable at new step
            strain_plas = strain_plas_n + gamma * n_tr  # Plastic strain at new step
            
            # Constants for computing algorithmic tangent stiffness
            beta_1 = 1 - ((yield_tr)/((norm_xi_tr)*(1 + (self.hardening / (3 * self.shear)))))
            beta_2 = (1 - (yield_tr)/(norm_xi_tr))*(1 + (self.hardening / (3 * self.shear)))
            
            # Computing deviatoric part of tangent stiffness
            dC = (2 * self.shear * beta_1 * P4sym) - (2 * self.shear * beta_2 * np.tensordot(n_tr,n_tr,axes=0))
            
            # Updating with spherical part of tangent stiffness
            C = dC + self.bulk * np.tensordot(I,I,axes=0)
            
            # Update stress by adding spherical and deviatoric part
            stress = dev_stress + self.bulk * np.trace(strain_new) * I
            
            # restore fourth order stiffness tensor as third order matrix
            C_red = tensor.fourth_to_three(C)
        # Restore stress into a vector
        stress_red = np.array([stress[0,0],stress[1,1],stress[0,1]])
        # Restore plastic strain into a vector
        strain_plas_red = np.array([strain_plas[0,0],strain_plas[1,1],2*strain_plas[0,1]])
        
        
        
        return stress_red, C_red, strain_plas_red, alpha  
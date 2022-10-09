# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
from input_abaqus import input_parameters
from tensor import tensor
inputs = input_parameters()
tensor = tensor()
class material_routine:
    def __init__(self,problem):
        """
        Caculates stiffness tensor for plane stress or plane strain condition.

        Parameters
        ----------
        Young : int
            Youngs modulus.
        Poisson : float
            Poissons ratio.

        Returns
        -------
        None.

        """
        if problem == 'Elastic':    
            self.k_const, self.G_c, self.l_0, self.Young, self.Poisson, self.stressState = inputs.material_parameters_elastic()
        elif problem == 'Elastic-Plastic':
            self.k_const, self.G_c, self.l_0, self.stressState, self.shear, self.bulk, self.sigma_y, self.hardening = inputs.material_parameters_elastic()
            self.Young = (9 * self.bulk * self.shear) / ((3 * self.bulk) + self.shear)
            self.Poisson = ((3 * self.bulk) - (2 * self.shear)) / ((6 * self.bulk) + (2 * self.shear))
    
    def planestress(self):  
        """
        Caculates stiffness tensor for plane stress condition.

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
        Caculates stiffness tensor for plane strain condition.

        Returns
        -------
        Ce : Array of float64, size(3,3)
            Stiffness tensor for plane straincase.

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
        Caculates stiffness tensor for plane strain condition.

        Returns
        -------
        Ce : Array of float64, size(3,3)
            Stiffness tensor for plane straincase.

        """
        if self.stressState == 1:  # Plane stress
            Ce = self.planestress()
        elif self.stressState == 2:  # Plane strain
            Ce = self.planestrain()
        
        return Ce
    def material_plasticity(self,strain, strain_plas, alpha):
        """
        Caculates stiffness tensor for plane strain condition.

        Returns
        -------
        Ce : Array of float64, size(3,3)
            Stiffness tensor for plane straincase.

        """
        tol = 1e-8
        P4sym = tensor.P4sym()
        I = np.eye(3)
        if self.stressState == 1:  # Plane stress
            Ce = self.planestress()
        elif self.stressState == 2:  # Plane strain
            Ce = self.planestrain()
        
        strain_new = np.array([[strain[0],strain[2],0],[strain[2],strain[1],0],[0,0,0]])
        strain_plas_n = np.array([strain_plas[0],strain_plas[2],0],[strain_plas[2],strain_plas[1],0],[0,0,0])
        alpha_n = alpha
        
        dev_strain = strain_new - (1 / 3) * np.trace(strain_new) * np.eye(3)
        dev_stress_tr = 2 * self.shear * (dev_strain - strain_plas_n)
        beta_tr = self.hardening * alpha_n
        xi_tr = dev_stress_tr
        norm_xi_tr = np.linalg.norm(xi_tr)
        n_tr = xi_tr / norm_xi_tr
        yield_tr = norm_xi_tr - np.sqrt(2/3)*(self.sigma_y + beta_tr)
        
        if yield_tr < tol:
            C = Ce
            strain_plas = strain_plas_n
            alpha = alpha_n
            stress = np.matmul(C,(strain_new - strain_plas_n))
        else:
            gamma = yield_tr / (2 * self.shear + (2 / 3)*self.hardening)
            dev_stress = dev_stress_tr - (2 * self.shear * gamma * n_tr)
            alpha = alpha_n + gamma * np.sqrt(2 / 3)
            strain_plas = strain_plas_n + gamma * n_tr
            beta_1 = 1 - ((yield_tr)/((norm_xi_tr)*(1 + (self.hardening / (3 * self.shear)))))
            beta_2 = (1 - (yield_tr)/(norm_xi_tr))*(1 + (self.hardening / (3 * self.shear)))
            dC = (2 * self.shear * beta_1 * P4sym) - (2 * self.shear * beta_2 * tensor.t2_otimes_t2(n_tr,n_tr))
            
            C = dC + self.bulk * tensor.t2_otimes_t2(I,I)
            stress = dev_stress + self.bulk * np.trace(strain_new) * I
        
        stress_red = np.array([stress[0,0],stress[1,1],stress[0,1]])
        strain_plas_red = np.array([strain_plas[0,0],strain_plas[1,1],strain_plas[0,1]])
        C_red = np.copy(C)
        C_red = np.delete(C_red,[2,4,5],axis = 0)
        C_red = np.delete(C_red,[2,4,5],axis = 1)
        
        return stress_red, C_red, strain_plas_red, alpha  
# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
from element_staggered import element_staggered
from assembly_index import assembly
class global_assembly():
    def __init__(self,num_elem,Points,Weights,C,disp,Phi,stress,strain,num_tot_var_u,num_tot_var_phi,num_dof_u,num_dof_phi):
        self.num_elem = num_elem
        self.Points = Points
        self.Weights = Weights
        self.C = C
        self.disp = disp
        self.Phi = Phi
        self.stress = stress
        self.strain = strain
        self.num_tot_var_u = num_tot_var_u
        self.num_tot_var_phi = num_tot_var_phi
        self.num_dof_u = num_dof_u
        self.num_dof_phi = num_dof_phi
        
    def global_disp(self):
        """
        

        Parameters
        ----------
        num_tot_var_u : TYPE
            DESCRIPTION.
        num_dof_u : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.global_K_disp = np.zeros((self.num_tot_var_u, self.num_tot_var_u))
        
        # Internal force vector
        self.global_F_int = np.zeros(self.num_tot_var_u)
        for elem in range(self.num_elem):
            elem_stiff = element_staggered(elem,self.Points,self.Weights,self.C,self.disp,self.Phi,self.stress,self.strain)
            F_int_elem = elem_stiff.element_internal_force()
            K_uu = elem_stiff.element_stiffness_displacement()
            assemble = assembly()
            
            index_u = assemble.assembly_index_u(elem,self.num_dof_u)
            
            X,Y = np.meshgrid(index_u,index_u,sparse=True)
            self.global_K_disp[X,Y] =  self.global_K_disp[X,Y] + K_uu
            self.global_F_int[index_u] = self.global_F_int[index_u]+F_int_elem
    def global_stiffness_disp(self):
        """
        

        Parameters
        ----------
        num_tot_var_u : TYPE
            DESCRIPTION.

        Returns
        -------
        global_K_disp : TYPE
            DESCRIPTION.

        """
        self.global_disp()
        return self.global_K_disp
    def global_internal_force(self):
        """
        

        Parameters
        ----------
        num_tot_var_u : TYPE
            DESCRIPTION.
        num_dof_u : TYPE
            DESCRIPTION.

        Returns
        -------
        global_K : TYPE
            DESCRIPTION.

        """
        self.global_disp()
        return self.global_F_int
    
    def global_phasefield(self):
        """
        

        Parameters
        ----------
        num_tot_var_phi : TYPE
            DESCRIPTION.
        num_dof_phi : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Initialization of global force vector and global stiffness matrix for phase field parameter
        self.global_residual_phi = np.zeros(self.num_tot_var_phi)
        self.global_K_phi = np.zeros((self.num_tot_var_phi, self.num_tot_var_phi))
             
        for elem in range(self.num_elem):
            elem_stiff = element_staggered(elem,self.Points,self.Weights,self.C,self.disp,self.Phi,self.stress,self.strain)
            K_phiphi,residual_phi,H_n = elem_stiff.element_stiffness_field_parameter()
            assemble = assembly()
            
            index_phi = assemble.assembly_index_phi(elem,self.num_dof_phi,self.num_tot_var_phi)
            
            X,Y = np.meshgrid(index_phi,index_phi,sparse=True)
            self.global_K_phi[X,Y] =  self.global_K_phi[X,Y] + K_phiphi
            self.global_residual_phi[index_phi] = self.global_residual_phi[index_phi]+residual_phi
            
    def global_stiffness_phasefield(self):
        """
        

        Parameters
        ----------
        num_tot_var_u : TYPE
            DESCRIPTION.

        Returns
        -------
        global_K_phi : TYPE
            DESCRIPTION.

        """
        self.global_phasefield()
        return self.global_K_phi
    def global_residual_phasefield(self):
        """
        

        Parameters
        ----------
        num_tot_var_u : TYPE
            DESCRIPTION.
        num_dof_u : TYPE
            DESCRIPTION.

        Returns
        -------
        global_K : TYPE
            DESCRIPTION.

        """
        self.global_phasefield()
        return self.global_residual_phi
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
    """
    Class to assemble element stiffness matrices and element residual vectors to global stiffness matrix
    and global residual vector respectively for both displacement and fracture phase field.
    I have excluded this functions from my routine, since it is taking more time to 
    execute, instead I have assemble directly in main file. 
    """
    def __init__(self,Points,Weights,disp,Phi,stress,strain,strain_energy,elements,nodes,num_dof,num_node_elem,num_Gauss_2D):
        """
        Assembled global stiffness matrix and residual vectors for both displacement and order parameter

        Parameters
        ----------
        Points : Array of float64, size(num_dim,num_Gauss**num_dim)
            Gauss points used for integration.
        Weights : Array of float64, size(num_Gauss**num_dim)
            Weights for Gauss points used for integration.
        disp : Array of float64, size(num_tot_var_u)
            Displacements.
        Phi : Array of float64, size(num_tot_var_phi)
            Phase field order parameter.
        stress : Array of float64, size(num_elem,num_Gauss_2D,num_stress)
            Stress at the integration points for all elements.
        strain : Array of float64, size(num_elem,num_Gauss_2D,num_stress)
            Strain at the integration points for all elements.
        strain_energy : Array of float64, size(num_elem,num_Gauss_2D)
            Strain energy at the integration points for all elements.
        elements : Array of int, size(num_elem,num_node_elem)
            Element connectivity matrix.
        nodes : Array of float64, size(num_node,num_dim)
            Co-ordinates of all field nodes.
        num_dof : int
            Total number of DOFs per node.
        num_node_elem : int
            Number of nodes per element.
        num_Gauss_2D : int
            Total number of Gauss points used for integration in 2-dimension.

        Returns
        -------
        None.

        """
        self.num_elem = len(elements)
        self.Points = Points
        self.Weights = Weights
        self.disp = disp
        self.Phi = Phi
        self.stress = stress
        self.strain = strain
        self.strain_energy = strain_energy
        self.elements = elements
        self.nodes = nodes
        self.num_dof = num_dof
        self.num_node_elem = num_node_elem
        self.num_Gauss_2D = num_Gauss_2D
    def global_disp(self,C,k_const):
        """
        Assembles global stiffness matrix and internal force vector for displacement

        Parameters
        ----------
        C : Array of float64, size(k)
            Stiffness tensor.
        k_const : float64
            parameter to avoid overflow for cracked elements.

        Returns
        -------
        None.

        """
        # the number of DOFs for displacements per node
        num_dof_u = self.num_dof - 1
        # total number of variables for displacements
        num_tot_var_u = len(self.nodes)*num_dof_u
        
        # Initialization of global internal force vector and global stiffness matrix for  displacement
        self.global_F_int = np.zeros(num_tot_var_u)
        self.global_K_disp = np.zeros((num_tot_var_u, num_tot_var_u))
        
        # loop for all elements
        for elem in range(self.num_elem):
            # Calling stiffness and internal force vector for an element 
            elem_stiff = element_staggered(elem,self.Points,self.Weights,self.disp,self.Phi,self.stress,self.strain,self.strain_energy,self.elements,self.nodes,self.num_dof,self.num_node_elem,self.num_Gauss_2D)
            F_int_elem = elem_stiff.element_internal_force(k_const)
            K_uu = elem_stiff.element_stiffness_displacement(C,k_const)
            
            # Calling assembly class and getting index for assembly
            assemble = assembly()
            index_u = assemble.assembly_index_u(elem,num_dof_u,self.num_node_elem,self.elements)
            
            X,Y = np.meshgrid(index_u,index_u,sparse=True)
            # Assemble global stiffness matrix
            self.global_K_disp[X,Y] =  self.global_K_disp[X,Y] + K_uu
            # Assemble global internal force vetor
            self.global_F_int[index_u] = self.global_F_int[index_u]+F_int_elem
    
    def global_phasefield(self,G_c,l_0):
        """
        Assembles global stiffness matrix and residual vector for phase field order parameter

        Parameters
        ----------
        G_c : float64
            Critical energy release for unstable crack or damage.
        l_0 : float64
            Length parameter which controls the spread of damage.

        Returns
        -------
        None.

        """
        num_dof_phi = self.num_dof - 2
        num_tot_var_phi = len(self.nodes)*num_dof_phi
        # Initialization of global force vector and global stiffness matrix for phase field parameter
        self.global_residual_phi = np.zeros(num_tot_var_phi)
        self.global_K_phi = np.zeros((num_tot_var_phi, num_tot_var_phi))
             
        for elem in range(self.num_elem):
            elem_stiff = element_staggered(elem,self.Points,self.Weights,self.disp,self.Phi,self.stress,self.strain,self.strain_energy,self.elements,self.nodes,self.num_dof,self.num_node_elem,self.num_Gauss_2D)
            K_phiphi = elem_stiff.element_stiffness_field_parameter(G_c,l_0)
            residual_phi = elem_stiff.element_residual_field_parameter(G_c,l_0)
            assemble = assembly()
            
            index_phi = assemble.assembly_index_phi(elem,num_dof_phi,num_tot_var_phi,self.num_node_elem,self.elements)
            
            X,Y = np.meshgrid(index_phi,index_phi,sparse=True)
            self.global_K_phi[X,Y] =  self.global_K_phi[X,Y] + K_phiphi
            self.global_residual_phi[index_phi] = self.global_residual_phi[index_phi]+residual_phi
    
    def global_stiffness_disp(self,C,k_const):
        """
        Call and returns global stiffness matrix for displacement

        Parameters
        ----------
        C : Array of float64, size(k)
            Stiffness tensor.
        k_const : float64
            parameter to avoid overflow for cracked elements.

        Returns
        -------
        Array of float64, size(num_tot_var_u,num_tot_var_u)
            global stiffness matrix.

        """
        self.global_disp(C,k_const)
        return self.global_K_disp
    
    def global_internal_force(self,C,k_const):
        """
        Call and returns global internal force vector for displacement

        Parameters
        ----------
        C : TYPE
            DESCRIPTION.
        k_const : TYPE
            DESCRIPTION.

        Returns
        -------
        Array of float64, size(num_tot_var_u)
            global stiffness internal force vector.

        """
        self.global_disp(C,k_const)
        return self.global_F_int
    
    def global_stiffness_phasefield(self,G_c,l_0):
        """
        Call and returns global stiffness matrix for phase field order parameter

        Parameters
        ----------
        G_c : float64
            Critical energy release for unstable crack or damage.
        l_0 : float64
            Length parameter which controls the spread of damage.

        Returns
        -------
        Array of float64, size(num_tot_var_phi,num_tot_var_phi)
            global stiffness matrix.

        """
        self.global_phasefield(G_c,l_0)
        return self.global_K_phi
    
    def global_residual_phasefield(self,G_c,l_0):
        """
        Call and returns global residual vector for displacement

        Parameters
        ----------
        G_c : float64
            Critical energy release for unstable crack or damage.
        l_0 : float64
            Length parameter which controls the spread of damage.

        Returns
        -------
        Array of float64, size(num_tot_var_phi)
            global residual vector.

        """
        self.global_phasefield(G_c,l_0)
        return self.global_residual_phi
# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
class assembly:
    """Class for computing indices to assemble global matrices."""
    def __init__(self):
        pass
    def assembly_index_u(self,elem:int,num_dof_u:int,num_node_elem:int,elements:np.array):
        """
        Function to generate a vector which consists indices of global matrix for 
        respective element belongs to displacement.

        Parameters
        ----------
        elem : int
            Element number.
        num_dof_u : int
            Number of degrees of freedom of displacement.
        num_node_elem : int
            Number of nodes per element, possible nodes are 2,3,4 and 8 per element.
        elements : Array of float64, size(num_elem,num_node_elem)
            Element connectivity matrix.
        Returns
        -------
        index : Array of float64, size(num_dof_elem)
            A vector which consists indices of global matrix for 
            respective element belongs to displacement.

        """
        num_elem_var_u = num_node_elem*num_dof_u # total number of element variables
        k = 0
        index = np.zeros(num_elem_var_u,int)
        for i in range(0,num_node_elem):
            temp = (elements[elem,i]-1)*num_dof_u
            for j in range(0,num_dof_u):
                index[k] = temp+j
                k = k+1
        return index
    
    def assembly_index_phi(self,elem:int,num_dof_phi:int,num_node_elem:int,elements:np.array):
        """
        Function to generate a vector which consists indices of global matrix for 
        respective element belongs to order parameter for staggered scheme.

        Parameters
        ----------
        elem : int
            Element number.
        num_dof_phi : int
            Number of degrees of freedom of order parameter.
        num_node_elem : int
            Number of nodes per element, possible nodes are 2,3,4 and 8 per element.
        elements : Array of float64, size(num_elem,num_node_elem)
            Element connectivity matrix.

        Returns
        -------
        index : Array of float64, size(num_dof_elem)
            A vector which consists indices of global matrix for 
            respective element belongs to order parameter.

        """
        num_elem_var_phi = num_node_elem*num_dof_phi # total number of element variables
        k = 0
        index = np.zeros(num_elem_var_phi,int)
        for i in range(0,num_node_elem):
            temp = (elements[elem,i]-1)*num_dof_phi
            for j in range(0,num_dof_phi):
                index[k] = temp+j
                k = k+1
        return index
    
    def assembly_index_phi_monolithic(self,elem:int,num_dof_phi:int,num_tot_var_u:int,num_node_elem:int,elements:np.array):
        """
        Function to generate a vector which consists indices of global matrix for 
        respective element belongs to order parameter for monolithic scheme.

        Parameters
        ----------
        elem : int
            Element number.
        num_dof_phi : int
            Number of degrees of freedom of order parameter.
        num_tot_var_u : int
            Total number of variables for displacements.
        num_node_elem : int
            Number of nodes per element, possible nodes are 2,3,4 and 8 per element.
        elements : Array of float64, size(num_elem,num_node_elem)
            Element connectivity matrix.

        Returns
        -------
        index : Array of float64, size(num_dof_elem)
            A vector which consists indices of global matrix 
            for respective element belongs to order parameter.

        """
        num_elem_var_phi = num_node_elem*num_dof_phi # total number of element variables
        k = 0
        index = np.zeros(num_elem_var_phi,int)
        for i in range(0,num_node_elem):
            temp = (elements[elem,i]-1)*num_dof_phi
            for j in range(0,num_dof_phi):
                index[k] = temp+j+num_tot_var_u
                k = k+1
        return index
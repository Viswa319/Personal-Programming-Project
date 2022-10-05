# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
# from input_parameters import *
import numpy as np
class assembly:
    def __init__(self):
        pass
    def assembly_index_u(self,elem,num_dof_u,num_node_elem,elements):
        """
        Function to generate a vector which consists indices of global matrix for respective element belongs to first field parameter. 
    
        Parameters
        ----------
        elements : Array of float64, size(num_elem,num_node_elem) 
            element connectivity matrix.
        elem : int
            element number.
        num_dof : int
            number of degrees of freedom.
        num_node_elem : int
                number of nodes per element, possible nodes are 2,3,4 and 8 per element.
    
        Returns
        -------
        index : Array of float64, size(num_dof_elem)
            a vector which consists indices of global matrix for respective element belongs to first field parameter.
    
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
    def assembly_index_phi(self,elem,num_dof_phi,num_tot_var_u,num_node_elem,elements):
        """
        Function to generate a vector which consists indices of global matrix for respective element belongs to second field parameter. 
    
        Parameters
        ----------
        elements : Array of float64, size(num_elem,num_node_elem) 
            element connectivity matrix.
        elem : int
            element number.
        num_dof : int
            number of degrees of freedom.
        num_node_elem : int
                number of nodes per element, possible nodes are 2,3,4 and 8 per element.
    
        Returns
        -------
        index : Array of float64, size(num_dof_elem)
            a vector which consists indices of global matrix for respective element belongs to second field parameter.
    
        """
        num_elem_var_phi = num_node_elem*num_dof_phi # total number of element variables
        k = 0
        index = np.zeros(num_elem_var_phi,int)
        for i in range(0,num_node_elem):
            temp = (elements[elem,i]-1)*num_dof_phi
            for j in range(0,num_dof_phi):
                # index[k] = temp+j+num_tot_var_u
                index[k] = temp+j
                k = k+1
        return index
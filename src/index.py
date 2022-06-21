# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
def assembly_index(elements,elem,num_dof,num_node_elem):
    """
    Function to generate a vector which consists indices of global matrix for respective element. 

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
        a vector which consists indices of global matrix for respective element.

    """
    num_elem_var = num_node_elem*num_dof # total number of element variables
    k = 0
    index = np.zeros(num_elem_var)
    for i in range(0,num_node_elem):
        temp = (elements[elem,i]-1)*num_dof
        for j in range(0,num_dof):
            index[k] = temp+j
            k = k+1
    return index
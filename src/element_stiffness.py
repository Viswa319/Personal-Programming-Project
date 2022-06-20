# # *************************************************************************
# #                 Implementation of phase-field model of ductile fracture
# #                      Upadhyayula Sai Viswanadha Sastry
# #                         Personal Programming Project
# #                               65130
# # *************************************************************************
import numpy as np
from shape_function import shape_function
from Bmatrix import Bmatrix
def element_stiffness(elem,num_node_elem,num_dim,num_dof,elements,nodes,num_Gauss,Points,Weights,C):
    """
    Function for calculating element stiffness matrix.

    Parameters
    ----------
    elem : int
        element number.
    num_node_elem : int
            number of nodes per element, possible nodes are 2,3,4 and 8 per element.
    num_dim : int
        dimension of a problem.
    num_dof : int
        number of degrees of freedom.
    elements : Array of float64, size(num_elem,num_node_elem) 
        element connectivity matrix.
    nodes : Array of float64, size(num_nodes,num_dim) 
        coordinated of nodes.
    num_Gauss : int
        number of Gauss points used for integration.
    Points :  Array of float64, size(num_dim,num_Gauss**num_dim)
        Gauss points used for integration.
    Weights : Array of float64, size(num_Gauss**num_dim)
        weights used for integration.
    C : Array of float64, size(3,3)
        Stiffness tensor.

    Raises
    ------
    ValueError
        Solution will be terminated if determinant of Jacobian is either zero or negative.

    Returns
    -------
    elem_K : Array of float64, size(num_elem_var,num_elem_var)
        element stiffness matrix.

    """
    num_elem_var = num_node_elem*num_dof
    # Initialization of element stiffness matrices
    elem_K = np.zeros((num_elem_var,num_elem_var))
   
    # Coordinates of nodes of current element
    elem_coord = np.zeros((num_node_elem,num_dim))
    for j in range(0,num_node_elem):
        elem_node = elements[elem,j]
        for k in range(0,num_dim):
            elem_coord[j,k] = nodes[elem_node-1,k]

    for j in range(0,num_Gauss**num_dim):
        gpos = Points[j]
        
        shapefunction = shape_function()
        if num_node_elem == 2:
            N,dNdxi = shapefunction.two_node_line_element(gpos)
        elif num_node_elem == 3:
            N,dNdxi = shapefunction.three_node_triangular_element(gpos)
        elif num_node_elem == 4:
            N,dNdxi = shapefunction.four_node_quadrilateral_element(gpos)
        elif num_node_elem == 8:
            N,dNdxi = shapefunction.eight_node_quadrilateral_element(gpos)
        
        Jacobian = np.matmul(dNdxi,elem_coord)
        det_Jacobian = np.linalg.det(Jacobian)
        if det_Jacobian <= 0:
            raise ValueError('Solution is terminated since, determinant of Jacobian is either zero or negative.')
        
        Jacobian_inv = np.linalg.inv(Jacobian)
        dNdX = np.matmul(Jacobian_inv,dNdxi)

        # Compute B matrix
        B = Bmatrix()
        Bmat = B.Bmatrix_linear(dNdX,num_node_elem)
        
        elem_K = elem_K + np.matmul(np.matmul(np.transpose(Bmat),C),Bmat)*Weights[j]*det_Jacobian
    return elem_K
# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
from shape_function import shape_function
from Bmatrix import Bmatrix
def solve_stress_strain(num_elem,num_Gauss_2D,num_stress,num_node_elem,num_dof_u,elements,disp,Points,nodes,C):
    num_elem_var_u = num_node_elem*num_dof_u # total number of displacements per element 
    stress = np.zeros((num_elem,num_Gauss_2D,num_stress))
    strain = np.zeros((num_elem,num_Gauss_2D,num_stress))
    elem_disp = np.zeros((num_elem,num_elem_var_u))
    
    for i in range(0,num_node_elem):
        elem_node = elements[:,i]
        for j in range(0,num_dof_u):
            i_elem_var = i*num_dof_u + j
            i_tot_var = (elem_node-1)*num_dof_u + j
            elem_disp[:,i_elem_var] = disp[i_tot_var]
    for elem in range(0,num_elem):
        elem_node_1 = elements[elem,:]
        elem_coord = nodes[elem_node_1-1,:]
        for j in range(0,num_Gauss_2D):
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
            Jacobian_inv = np.linalg.inv(Jacobian)
            dNdX = np.matmul(Jacobian_inv,dNdxi)
               
            # Compute B matrix
            B = Bmatrix()
            Bmat = B.Bmatrix_linear(dNdX,num_node_elem)
               
            strain[elem,j,:] = strain[elem,j,:] + np.matmul(Bmat,elem_disp[elem])
               
            stress[elem,j,:] = stress[elem,j,:] + np.matmul(C,strain[elem,j,:])
    return stress, strain
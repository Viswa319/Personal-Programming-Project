# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
from shape_function import shape_function
from Bmatrix import Bmatrix
import numpy as np
from input_parameters import *
from assembly_index import assembly
def residual(num_node,elem,num_node_elem,num_dim,num_dof,elements,nodes,num_Gauss,Points,Weights,C,disp,disp_old,stress,strain,global_force):
    for elem in range(0,num_elem):
    
        num_dof_u = num_dof-1
    
        # the number of DOFs for order parameters per node
        num_dof_phi = num_dof-2
        
        # total number of variables for displacements
        num_tot_var_u = num_node*num_dof_u
        
        # total number of displacements per element 
        num_elem_var_u = num_node_elem*num_dof_u
            
        # total number of order parameters per element 
        num_elem_var_phi = num_node_elem*num_dof_phi
            
        # Initialize element load vectors    
        R_elem_u = np.zeros((num_elem_var_u))
        R_elem_phi = np.zeros((num_elem_var_phi))
                   
        elem_node = elements[elem,:]
        i_tot_var = num_tot_var_u + elem_node
        elem_phi = disp[i_tot_var-1]
        elem_phi_dot = disp[i_tot_var-1] - disp_old[i_tot_var-1]
    
        elem_coord = nodes[elem_node-1,:]
        for j in range(0,num_Gauss**num_dim):
            gpos = Points[j]
            
            shape = shape_function(num_node_elem,gpos,elem_coord)
            N = shape.get_shape_function()
            dNdX = shape.get_shape_function_derivative()
            det_Jacobian = shape.get_det_Jacobian()
            if det_Jacobian <= 0:
                raise ValueError('Solution is terminated since, determinant of Jacobian is either zero or negative.')

            phi = np.matmul(elem_phi,N[0])
            phi_dot = np.matmul(elem_phi_dot,N[0])
                    
            phi_func = 0
            if phi_dot < 0:
                phi_func = -phi_dot
                    
            # Compute B matrix
            B = Bmatrix()
            Bmat = B.Bmatrix_linear(dNdX,num_node_elem)
            strain_energy = 0.5*np.dot(stress[elem,j,:],strain[elem,j,:])
            R_elem_u = R_elem_u + np.matmul(np.transpose(Bmat),stress[elem,j,:])*(((1-phi)**2)+k_const)*Weights[j]*det_Jacobian
                
            R_elem_phi = R_elem_phi + G_c*l_0*np.matmul(np.transpose(dNdX),np.matmul(dNdX,np.array([phi,phi,phi,phi])))*Weights[j]*det_Jacobian
            R_elem_phi = R_elem_phi + (((G_c/l_0)+2*strain_energy)*N[0]*phi-2*N[0]*(strain_energy-(neta/delta_time)*phi_func))*Weights[j]*det_Jacobian
                
        assemble = assembly()
        
        index_u = assemble.assembly_index_u(elem,num_dof_u,num_node_elem,elements)
        index_phi = assemble.assembly_index_phi(elem,num_dof_phi,num_tot_var_u,num_node_elem,elements)
        
        global_force[index_u] = global_force[index_u] + R_elem_u
        global_force[index_phi] = global_force[index_phi] + R_elem_phi
    return global_force
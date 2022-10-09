# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
from shape_function import shape_function
from Bmatrix import Bmatrix
from input_parameters import *
class element_stiffness():
    def __init__(self):
        pass
    def element_stiffness_monolithic(self,elem,Points,Weights,C,disp,disp_old,stress,strain):
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
        num_dof_u = num_dof-1
        
        # the number of DOFs for order parameters per node
        num_dof_phi = num_dof-2
        
        # total number of variables for displacements
        num_tot_var_u = num_node*num_dof_u
        
        # total number of displacements per element 
        num_elem_var_u = num_node_elem*num_dof_u
        
        # total number of order parameters per element 
        num_elem_var_phi = num_node_elem*num_dof_phi
        # Initialize element stiffness matrices    
        K_uu = np.zeros((num_elem_var_u,num_elem_var_u))
        K_uphi = np.zeros((num_elem_var_u,num_elem_var_phi))
        K_phiu = np.zeros((num_elem_var_phi,num_elem_var_u))
        K_phiphi = np.zeros((num_elem_var_phi,num_elem_var_phi))
       
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
            B = Bmatrix(dNdX,num_node_elem)
            Bmat = B.Bmatrix_disp()
            
            K_uu = K_uu + np.matmul(np.matmul(np.transpose(Bmat),C),Bmat)*(((1-phi)**2)+k_const)*Weights[j]*det_Jacobian
                
            K_uphi = K_uphi -2*(1-phi)*np.matmul(np.transpose(Bmat),np.transpose(stress[elem])*N[0])*Weights[j]*det_Jacobian
            K_phiu = np.transpose(K_uphi)
            # K_phiu = K_phiu -2*(1-phi)*np.matmul(stress[elem],Bmat)*N[0,j]*Weights[j]*det_Jacobian
            
            strain_energy = 0.5*np.dot(stress[elem,j,:],strain[elem,j,:])
            K_phiphi = K_phiphi + G_c*l_0*np.matmul(np.transpose(dNdX),dNdX)*Weights[j]*det_Jacobian
            
            K_phiphi = K_phiphi + (((G_c/l_0)+2*strain_energy)+(neta/delta_time)*phi_func)*np.matmul(np.transpose([N[0]]),[N[0]])*Weights[j]*det_Jacobian
        return K_uu,K_uphi,K_phiu,K_phiphi

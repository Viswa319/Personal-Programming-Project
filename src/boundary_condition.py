# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np


# def boundary_condition(num_dof_u,fixed_dof,global_K,global_force,disp,disp_bc):
#     """
#     Apply essential boundary conditions

#     Parameters
#     ----------
#     num_dof_u : int
#         Total number of DOFs for displacement per node.
#     fixed_dof : Array of float64, size(num_tot_var_u)
#         If boundary condition is prescribed at certain node, then it is given as 1 or else 0.
#     global_K : Array of float64, size(num_tot_var_u,num_tot_var_u)
#         Global stiffness matrix.
#     global_force : Array of float64, size(num_tot_var_u)
#         Global stiffness internal force vector.
#     disp : Array of float64, size(num_tot_var_u)
#         Displacements.
#     disp_bc : Array of float64, size(num_tot_var_u)
#         Displacements at boundary.

#     Returns
#     -------
#     global_K : Array of float64, size(num_tot_var_u,num_tot_var_u)
#         Global stiffness matrix.
#     global_force : Array of float64, size(num_tot_var_u)
#         Global stiffness internal force vector.

#     """
#     num_tot_var = len(disp)
#     for i in range(num_tot_var):
#         if fixed_dof[i] == 1:
#             global_K[i,:] = 0.0
#             global_K[i,i] = 1.0

#             global_force[i] = disp_bc[i]
#     return global_K,global_force


# def boundary_condition(num_dof_u,fixed_dof,global_K,global_force,disp,disp_bc):
#     num_tot_var = len(disp)
#     num_fixed_dof = len(np.where(fixed_dof != 0)[0])
#     global_K_reduced = np.zeros((num_tot_var-num_fixed_dof,num_tot_var-num_fixed_dof))
#     global_force_reduced = np.zeros(num_tot_var-num_fixed_dof)
#     k = 0
#     for i in range(num_tot_var):
#         l = 0
#         if fixed_dof[i] != 1:
#             for j in range(num_tot_var):
#                 if fixed_dof[j] != 1:
#                     global_K_reduced[k,l] = global_K[i,j]
#                     l = l+1
#             global_force_reduced[k] = global_force[i]
#             k = k+1
#     return global_K_reduced,global_force_reduced

def boundary_condition(num_dof_u,nodes_bc,fixed_dof,global_K,global_force,disp,disp_bc,tot_inc):
    num_fixnodes = len(nodes_bc)
    num_tot_var = len(disp)
    external_force = np.zeros((num_fixnodes,num_dof_u))
    for i in range(0,num_fixnodes):
        lnode = nodes_bc[i]
        for j in range(0,num_dof_u):
            if fixed_dof[i,j] == 1:
                itotv = ((lnode-1)*num_dof_u+j)
                external_force[i,j] = external_force[i,j]-np.dot(global_K[itotv,:],disp[0:num_tot_var])
                
                global_K[itotv,:] = 0.0
                global_K[itotv,itotv] = 1.0
                
                global_force[itotv] = disp_bc[i,j]*tot_inc-disp[itotv]
    return global_K,global_force,external_force

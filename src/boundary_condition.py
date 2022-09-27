# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
def boundary_condition(num_dof_u,nodes_bc,fixed_dof,global_K,global_force,disp,disp_bc,tot_inc):
    num_fixnodes = len(nodes_bc)
    num_tot_var = len(disp)
    for i in range(num_fixnodes):
        lnode = nodes_bc[i]
        for j in range(num_dof_u):
            if fixed_dof[i,j] == 1:
                itotv = ((lnode-1)*num_dof_u+j)
                global_K[itotv,:] = 0.0
                
                global_K[itotv,itotv] = 1.0
                
                global_force[itotv] = disp_bc[i,j]*tot_inc+disp[itotv]
    return global_K,global_force
# def boundary_condition(num_dof_u,fixed_dof,global_K,global_force,disp,disp_bc,tot_inc):
#     # num_fixnodes = len(nodes_bc)
#     num_tot_var = len(disp)
#     for i in range(num_tot_var):
#         if fixed_dof[i] == 1:
#             for j in range(num_tot_var):
#                 if fixed_dof[j] == 0:
#                     global_force[j] = global_force[j] - global_K[j,i] * disp_bc[i]
#             global_K[i,:] = 0.0
#             global_K[:,i] = 0.0
#             global_K[i,i] = 1.0
#             global_force[i] = disp_bc[i]
#     return global_K,global_force
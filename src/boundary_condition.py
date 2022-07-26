# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
def boundary_condition(num_dof_u,nodes_bc,fixed_dof,global_K,global_force,disp,disp_bc,tot_inc):
    # num_fixnodes = len(nodes_bc)
    # num_tot_var = len(disp)
    # external_force = np.zeros((num_fixnodes,num_dof_u))
    
    # for i in range(0,num_dof_u):
    #     ind = np.where(fixed_dof[:,i] == 1)[0]
    #     nodes = nodes_bc[ind]
    #     itotv = (nodes-1)*num_dof_u+i
    #     external_force[ind,i] = external_force[ind,i]-np.dot(global_K[itotv,:],disp[0:num_tot_var])
    #     global_force[itotv] = disp_bc[ind,i]*tot_inc-disp[itotv]
    # global_K[itotv,:] = 0.0
    # global_K[itotv,itotv] = 1.0
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
    return(global_K,global_force,external_force)
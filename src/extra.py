# def shape_function(point,num_node_elem):
#     """
#     Function to get the Lagrange interpolant basis and its gradients w.r.t its coordinates.

#     Parameters
#     ----------
#     point : array of float64
#         point.
#     num_node_elem : int
#         number of nodes per element, possible nodes are 2,3,4 and 8 per element.
    
#     Returns
#     -------
#     N : array of float64
#         Lagrange interpolant shape function.
#     dNdxi : array of float64
#         gradient of Lagrange interpolant shape function w.r.t respecive coordinates.
#     """
#     if num_node_elem == 2:
#         ########### two node line element ########## 
#         #  
#         #    1---------2
#         #
#         # Shape functions
#         N = np.array([(1+point)/2,(1-point)/2])[np.newaxis] 
        
#         # derivatives of shape functions w.r.t psi
#         dNdxi = np.array([1/2,-1/2])[np.newaxis]
#     elif num_node_elem == 3:
#         ########### three node triangular element ########## 
#         #  
#         #    1---------2
#         #     \       /
#         #      \     /
#         #       \   /
#         #        \ /
#         #         3
#         #
#         # Shape functions
#         eta = point[1]
#         psi = point[0]
#         N = np.array([1-psi-eta,psi,eta])[np.newaxis] 
        
#         # derivatives of shape functions w.r.t psi and eta respectively
#         dNdpsi = np.array([-1, 1, 0])[np.newaxis]
#         dNdeta = np.array([-1, 0, 1])[np.newaxis]
#         dNdxi = np.r_[dNdpsi,dNdeta]
#     elif num_node_elem == 4:
#         ########### four node quadrilateral element ########## 
#         #  
#         #    4---------3
#         #    |         | 
#         #    |         |
#         #    |         |
#         #    1---------2
#         # Shape functions
#         eta = point[1]
#         psi = point[0]
#         N = 0.25*np.array([(1-psi)*(1-eta), (1+psi)*(1-eta), (1+psi)*(1+eta), (1-psi)*(1+eta)])[np.newaxis]
    
#         # derivatives of shape functions w.r.t psi and eta respectively
#         dNdpsi = 0.25*np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])[np.newaxis]
#         dNdeta = 0.25*np.array([-(1-psi), -(1+psi), (1+psi), (1-psi)])[np.newaxis]
#         dNdxi = np.r_[dNdpsi,dNdeta]
#     elif num_node_elem == 8:
#         ########### eight node quadrilateral element ########## 
#         #
#         #    7-----6-----5
#         #    |           | 
#         #    |           |
#         #    8           4 
#         #    |           |
#         #    |           |
#         #    1-----2-----3
#         # Shape functions
#         eta = point[1]
#         psi = point[0]
#         N = np.array([-1*0.25*(1-psi)*(1-eta)*(1+psi+eta), 0.5*(1-psi)*(1-eta)*(1+psi), -1*0.25*(1+psi)*(1-eta)*(1-psi+eta), 0.5*(1+psi)*(1-eta)*(1+eta), -1*0.25*(1+psi)*(1+eta)*(1-psi-eta), 0.5*(1+psi)*(1+eta)*(1-psi),-1*0.25*(1-psi)*(1+eta)*(1+psi-eta), 0.5*(1-psi)*(1+eta)*(1-eta)])[np.newaxis]
        
#         # derivatives of shape functions w.r.t psi and eta respectively
#         dNdpsi = np.array([0.25*(1-eta)*(2*psi+eta), -psi*(1-eta), 0.25*(1-eta)*(2*psi-eta), 0.5*(1-eta)*(1+eta), 0.25*(1+eta)*(2*psi+eta),-psi*(1+eta),0.25*(1+eta)*(2*psi-eta),-0.5*(1-eta)*(1+eta)])[np.newaxis]
#         dNdeta = np.array([0.25*(1-psi)*(psi+2*eta), -0.5*(1-psi)*(1+psi), 0.25*(1+psi)*(-psi+2*eta), -eta*(1+psi), 0.25*(1+psi)*(psi+2*eta),0.5*(1-psi)*(1+psi),0.25*(1-psi)*(-psi+2*eta),-eta*(1-psi)])[np.newaxis]
#         dNdxi = np.r_[dNdpsi,dNdeta]
#     else:
#         raise ValueError("Number of nodes per element provided is not possible. Possible nodes per element are 2,3,4 and 8")
#     return N,dNdxi

 # elem_phi = np.zeros(num_node_elem)
        # elem_phi_dot = np.zeros(num_node_elem)
        
        # # coordinates of nodes which belongs to particular element
        # elem_coord = np.zeros((num_node_elem,num_dim))
        # for i in range(0,num_node_elem):
        #     elem_node = elements[elem,i]
        #     i_tot_var = num_tot_var_u + elem_node
        #     elem_phi[i] = disp[i_tot_var-1]
        #     elem_phi_dot[i] = disp[i_tot_var-1] - disp_old[i_tot_var-1]
    
        #     for j in range(0,num_dim):
        #         elem_coord[i,j] = nodes[elem_node-1,j]

# for ievab in range(0,num_elem_var_phi):
#                 for jevab in range(0,num_elem_var_phi):
#                     for idim in range(0,num_dim):
#                         # K_phiphi[ievab,jevab] = K_phiphi[ievab,jevab] + dNdX[idim,ievab]*dNdX[idim,jevab]
#                         K_phiphi[ievab,jevab] = K_phiphi[ievab,jevab] + G_c*l_0*dNdX[idim,ievab]*dNdX[idim,jevab]*Weights[j]*det_Jacobian

# for innode in range(0,num_node_elem):
#                 for jnnode in range(0,num_node_elem):
            #         K_phiphi_2[innode,jnnode] = K_phiphi_2[innode,jnnode] + (((G_c/l_0)+2*strain_energy)+(neta/delta_time)*phi_func)*N[0,innode]*N[0,jnnode]*Weights[j]*det_Jacobian
            
    # num_fixnodes = len(fixnodes)
    # num_tot_var = len(disp)
    # external_force = np.zeros((num_fixnodes,num_dof_u))
    # for i in range(0,num_fixnodes):
    #     lnode = fixnodes[i]
    #     for j in range(0,num_dof_u):
    #         if fixdof[i,j] == 1:
    #             itotv = ((lnode-1)*num_dof_u+j)
    #             external_force[i,j] = external_force[i,j]-np.dot(global_K[itotv,:],disp[0:num_tot_var])
                
    #             global_K[itotv,:] = 0.0
                
    #             global_K[itotv,itotv] = 1.0
                
    #             global_force[itotv] = disp_bc[i,j]*tot_inc-disp[itotv]
    
    
# num_dof_u = num_dof-1
    
#     # the number of DOFs for order parameters per node
#     num_dof_phi = num_dof-2
        
#     # total number of variables for displacements
#     num_tot_var_u = num_node*num_dof_u
        
#     # total number of displacements per element 
#     num_elem_var_u = num_node_elem*num_dof_u
        
#     # total number of order parameters per element 
#     num_elem_var_phi = num_node_elem*num_dof_phi
#     K_uu_1 = np.zeros((num_elem,num_elem_var_u,num_elem_var_u))
#     K_uphi_1 = np.zeros((num_elem,num_elem_var_u,num_elem_var_phi))
#     K_phiu_1 = np.zeros((num_elem,num_elem_var_phi,num_elem_var_u))
#     K_phiphi_1 = np.zeros((num_elem,num_elem_var_phi,num_elem_var_phi))
    
#     elem_phi_1 = np.zeros((num_elem,num_node_elem))
#     elem_phi_dot_1 = np.zeros((num_elem,num_node_elem))
    
#     for i in range(0,num_node_elem):
#         elem_node_1 = elements[:,i]
#         i_tot_var = num_tot_var_u + elem_node_1
#         elem_phi_1[:,i] = disp[i_tot_var-1]
#         elem_phi_dot_1[:,i] = disp[i_tot_var-1] - disp_old[i_tot_var-1]
        
#     Weights_1 = []
#     det_Jacobian_1 = []
#     Bmat_u = np.zeros((num_elem,num_stress,num_elem_var_u))
#     strain_energy_1 = []
#     dVol = np.zeros((num_elem,num_Gauss**num_dim))
#     for elem in range(0,num_elem):
#         # K_uu,K_uphi,K_phiu,K_phiphi = element_stiffness(num_node,elem,num_node_elem,num_dim,num_dof,elements,nodes,num_Gauss,Points,Weights,C,disp,disp_old,stress,strain)
        
#         # Initialize element stiffness matrices    
#         K_uu = np.zeros((num_elem_var_u,num_elem_var_u))
#         K_uphi = np.zeros((num_elem_var_u,num_elem_var_phi))
#         K_phiu = np.zeros((num_elem_var_phi,num_elem_var_u))
#         K_phiphi = np.zeros((num_elem_var_phi,num_elem_var_phi))
        
        
#         elem_node = elements[elem,:]
#         i_tot_var = num_tot_var_u + elem_node
#         elem_phi = disp[i_tot_var-1]
#         elem_phi_dot = disp[i_tot_var-1] - disp_old[i_tot_var-1]
    
#         elem_coord = nodes[elem_node-1,:]
#         for j in range(0,num_Gauss**num_dim):
#             gpos = Points[j]
            
#             shapefunction = shape_function()
#             if num_node_elem == 2:
#                 N,dNdxi = shapefunction.two_node_line_element(gpos)
#             elif num_node_elem == 3:
#                 N,dNdxi = shapefunction.three_node_triangular_element(gpos)
#             elif num_node_elem == 4:
#                 N,dNdxi = shapefunction.four_node_quadrilateral_element(gpos)
#             elif num_node_elem == 8:
#                 N,dNdxi = shapefunction.eight_node_quadrilateral_element(gpos)
                
#             Jacobian = np.matmul(dNdxi,elem_coord)
#             det_Jacobian = np.linalg.det(Jacobian)
#             if det_Jacobian <= 0:
#                 raise ValueError('Solution is terminated since, determinant of Jacobian is either zero or negative.')
               
#             Jacobian_inv = np.linalg.inv(Jacobian)
#             dNdX = np.matmul(Jacobian_inv,dNdxi)
            
#             phi_1 = np.matmul(elem_phi_1,N[0])
#             phi_dot_1 = np.matmul(elem_phi_dot_1,N[0])
#             phi = np.matmul(elem_phi,N[0])
#             phi_dot = np.matmul(elem_phi_dot,N[0])
            
#             dVol[elem,j] = det_Jacobian*Weights[j]
#             phi_func = 0
#             if phi_dot < 0:
#                 phi_func = -phi_dot
            
#             Weights_1.append(Weights[j])
#             det_Jacobian_1.append(det_Jacobian)
#             # Compute B matrix
#             B = Bmatrix()
#             Bmat = B.Bmatrix_linear(dNdX,num_node_elem)
#             Bmat_u[elem,:,:] = Bmat[:,:]
            
#             K_uu_1[elem] = K_uu_1[elem] + np.matmul(np.matmul(np.transpose(Bmat),C),Bmat)*(((1-phi)**2)+k_const)*Weights[j]*det_Jacobian
            
#             K_uu = K_uu + np.matmul(np.matmul(np.transpose(Bmat),C),Bmat)*(((1-phi)**2)+k_const)*Weights[j]*det_Jacobian
            
#             K_uphi_1[elem] = K_uphi_1[elem] -2*(1-phi)*np.matmul(np.transpose(Bmat),np.transpose(stress[elem]))*N[0,j]*Weights[j]*det_Jacobian
#             K_uphi = K_uphi -2*(1-phi)*np.matmul(np.transpose(Bmat),np.transpose(stress[elem]))*N[0,j]*Weights[j]*det_Jacobian
            
#             K_phiu = K_phiu -2*(1-phi)*np.matmul(stress[elem],Bmat)*N[0,j]*Weights[j]*det_Jacobian
#             K_phiu_1[elem] = K_phiu_1[elem] -2*(1-phi)*np.matmul(stress[elem],Bmat)*N[0,j]*Weights[j]*det_Jacobian
#             strain_energy = 0.5*np.dot(stress[elem,j,:],strain[elem,j,:])
#             K_phiphi = K_phiphi + G_c*l_0*np.matmul(np.transpose(dNdX),dNdX)*Weights[j]*det_Jacobian
            
#             K_phiphi = K_phiphi + (((G_c/l_0)+2*strain_energy)+(neta/delta_time)*phi_func)*np.matmul(np.transpose([N[0]]),np.transpose(np.transpose([N[0]])))*Weights[j]*det_Jacobian
            
#             K_phiphi_1[elem] = K_phiphi_1[elem] + G_c*l_0*np.matmul(np.transpose(dNdX),dNdX)*Weights[j]*det_Jacobian
            
#             K_phiphi_1[elem] = K_phiphi_1[elem] + (((G_c/l_0)+2*strain_energy)+(neta/delta_time)*phi_func)*np.matmul(np.transpose([N[0]]),np.transpose(np.transpose([N[0]])))*Weights[j]*det_Jacobian
#             if np.array_equiv(np.round(K_uu,6),np.round(K_uu_1[elem],6)) is False:
#                 print('hey1')
#             if np.array_equiv(np.round(K_uphi,6),np.round(K_uphi_1[elem],6)) is False:
#                 print('hey2')
#             if np.array_equiv(np.round(K_phiu,6),np.round(K_phiu_1[elem],6)) is False:
#                 print('hey3')
#             if np.array_equiv(np.round(K_phiphi,6),np.round(K_phiphi_1[elem],6)) is False:
#                 print('hey4')
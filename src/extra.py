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

# for i in range(len(nodes_bc)):
#     lnode = nodes_bc[i]
#     for j in range(num_dof_u):
#         if fixed_dof[i,j] == 1:
#             itotv = ((lnode-1)*num_dof_u+j)
#             R_ext[itotv] = R_ext[itotv]-np.dot(global_K_disp[itotv,:],disp[0:num_tot_var])

# fixed_dof = np.zeros(num_tot_var_u)
# fixed_dof[(top*2)+1] = 1
# fixed_dof[bot*2] = 1
# fixed_dof[bot*2+1] = 1

# disp_bc = np.zeros(num_tot_var_u)
# disp_bc[top*2+1] = 0.1

# disp_bc_new += (1/num_step)*disp_bc
# disp_bc[top*2+1] = tot_inc

# global_force_disp = F_ext + R_ext - F_int

#tolerance = np.linalg.norm(global_force_disp)/np.linalg.norm(F_ext + R_ext)

# # Getting the nodes near crack
# crack = []
# for i in range(0,num_node):
#     if 0 <= nodes[i,0] and nodes[i,1] == 0:
#         crack.append(i)

# # assigning order parameter values as 1
# for i in range(0,len(crack)):
#     phi[crack[i]] = 1

# disp_bot = np.zeros((len(bot),num_dim))
# disp_top = np.zeros((len(top),num_dim))
# disp_right = np.zeros((len(bot),num_dim))
# disp_left = np.zeros((len(top),num_dim))

# fixed_dof_bot = np.zeros((len(bot),num_dim),int) # fixed degrees of freedom for each node
# fixed_dof_top = np.zeros((len(top),num_dim),int) # fixed degrees of freedom for each node
# fixed_dof_left = np.zeros((len(left),num_dim),int) # fixed degrees of freedom for each node
# fixed_dof_right = np.zeros((len(right),num_dim),int) # fixed degrees of freedom for each node
# fixed_dof_bot[:,0] = 1
# fixed_dof_bot[:,1] = 1
# fixed_dof_top[:,1] = 1
# fixed_dof_top[:,0] = 1
# fixed_dof_right[:,1] = 1
# fixed_dof_right[:,1] = 1
# fixed_dof_left[:,1] = 1
# fixed_dof_left[:,1] = 1
# fixed_dof = np.r_[fixed_dof_bot,fixed_dof_top,fixed_dof_right,fixed_dof_left]
# disp_bc = np.r_[disp_bot,disp_top,disp_right,disp_left]

# fixed_dof[left*2] = 1
# fixed_dof[left*2+1] = 1
# fixed_dof[right*2] = 1
# fixed_dof[right*2+1] = 1

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

            # self.strain_energy_new[self.elem,j] = 0.5*np.dot(self.stress[self.elem,j,:],self.strain[self.elem,j,:])

            # if self.strain_energy_new[self.elem,j] > self.strain_energy[self.elem,j]:
            #     H = self.strain_energy_new[self.elem,j]
            # else:
            #     H = self.strain_energy[self.elem,j]

# k = 0
        # for i in range(num_tot_var_u):
        #     if fixed_dof[i] != 1:
        #         disp[i] = disp_reduced[k]
        #         k = k+1
        #     if fixed_dof[i] == 1:
        #         disp[i] = disp_bc[i]

# step = 0
# while tot_inc <= 0.01:
    # step = step+1
    # if tot_inc <= 0.005:
    #     disp_inc = 1e-5
    # else:
    #     disp_inc = 1e-6

# if (np.linalg.norm(delta_phi) < max((10**-5)*np.linalg.norm(phi),10**-8)):
        #     print("Phase field solution converged!")
        #     break
        # if (np.linalg.norm(global_force_phi) < 0.005*max(np.linalg.norm(global_force_phi_tol),10**-8)):
        #     print("Displacement solution converged!")
        #     break

# Initialization of global force vector and global stiffness matrix for  displacement
    # global_force_disp = np.zeros(num_tot_var_u)
    # global_K_disp = np.zeros((num_tot_var_u, num_tot_var_u))

# for elem in range(0,num_elem):
    #     elem_stiff = element_staggered(elem,Points,Weights,disp,phi,stress,strain,strain_energy,elements,nodes,num_dof,num_node_elem,num_Gauss_2D)
    #     K_uu = elem_stiff.element_stiffness_displacement(C,k_const)
    #     assemble = assembly()

    #     index_u = assemble.assembly_index_u(elem,num_dof_u,num_node_elem,elements)

    #     X,Y = np.meshgrid(index_u,index_u,sparse=True)
    #     global_K_disp[X,Y] =  global_K_disp[X,Y] + K_uu
    #     del X,Y

# for elem in range(0,num_elem):
        #     elem_stiff = element_staggered(elem,Points,Weights,disp,phi,stress,strain,strain_energy,elements,nodes,num_dof,num_node_elem,num_Gauss_2D)
        #     F_int_elem = elem_stiff.element_internal_force(k_const)

        #     assemble = assembly()

        #     index_u = assemble.assembly_index_u(elem,num_dof_u,num_node_elem,elements)

        #     F_int[index_u] = F_int[index_u]+F_int_elem

# for elem in range(num_elem):
    #     elem_stiff = element_staggered(elem,Points,Weights,disp,phi,stress,strain,strain_energy,elements,nodes,num_dof,num_node_elem,num_Gauss_2D)
    #     K_phiphi = elem_stiff.element_stiffness_field_parameter(G_c,l_0)
    #     residual_elem_phi = elem_stiff.element_residual_field_parameter(G_c,l_0)
    #     assemble = assembly()

    #     index_phi = assemble.assembly_index_phi(elem,num_dof_phi,num_tot_var_phi,num_node_elem,elements)

    #     X,Y = np.meshgrid(index_phi,index_phi,sparse=True)
    #     global_K_phi[X,Y] =  global_K_phi[X,Y] + K_phiphi
    #     global_force_phi[index_phi] = global_force_phi[index_phi]+residual_elem_phi
    #     del X,Y

# Initialization of global force vector and global stiffness matrix for phase field parameter
    # global_force_phi = np.zeros(num_tot_var_phi)
    # global_K_phi = np.zeros((num_tot_var_phi, num_tot_var_phi))
    # global_force_phi_tol = np.zeros(num_tot_var_phi)

# for elem in range(num_elem):
        #     elem_stiff = element_staggered(elem,Points,Weights,disp,phi,stress,strain,strain_energy,elements,nodes,num_dof,num_node_elem,num_Gauss_2D)
        #     residual_elem_phi = elem_stiff.element_residual_field_parameter(G_c,l_0)
        #     assemble = assembly()

        #     index_phi = assemble.assembly_index_phi(elem,num_dof_phi,num_tot_var_phi,num_node_elem,elements)

        #     global_force_phi_tol[index_phi] = global_force_phi_tol[index_phi]+residual_elem_phi

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

# def boundary_condition(num_dof_u,nodes_bc,fixed_dof,global_K,global_force,disp,disp_bc,tot_inc):
#     num_fixnodes = len(nodes_bc)
#     num_tot_var = len(disp)
#     external_force = np.zeros((num_fixnodes,num_dof_u))
#     for i in range(0,num_fixnodes):
#         lnode = nodes_bc[i]
#         for j in range(0,num_dof_u):
#             if fixed_dof[i,j] == 1:
#                 itotv = ((lnode-1)*num_dof_u+j)
#                 external_force[i,j] = external_force[i,j]-np.dot(global_K[itotv,:],disp[0:num_tot_var])

#                 global_K[itotv,:] = 0.0
#                 global_K[itotv,itotv] = 1.0

#                 global_force[itotv] = disp_bc[i,j]*tot_inc-disp[itotv]
#     return global_K,global_force,external_force

# if np.mod(step,num_print-1) == 0:
    #     deflection = np.zeros((num_node,num_dim))
    #     order_parameter = np.zeros(num_node)
    #     for i in range(0,num_node):
    #         for j in range(0,num_dof_u):
    #             itotv = i*num_dof+j
    #             deflection[i,j] = nodes[i,j]+10*disp[itotv]
    #             jtotv = num_tot_var_u+i
    #             order_parameter[i] = disp[jtotv]
    #     file_name = 'time_{}.vtk'.format(step)
    #     vtk_generator(file_name,deflection,order_parameter)
    
# def test_essential_boundary_conditions_true():
#     '''
#     UNIT TESTING
#     Aim: Test essential boundary conditions

#     Expected result : Array of global stiffness matrix after applying boundary conditions

#     Test command : pytest test.py::test_essential_boundary_conditions_true()

#     Remarks : test case passed successfully
#     '''
#     actual_global_K_disp,actual_global_force_disp = boundary_condition(num_dof_u,fixed_dof,global_K_disp,global_force_disp,disp,disp_bc)
#     expected_global_K_disp = np.eye(num_tot_var_u)
#     assert(array_equiv(np.round(actual_global_K_disp,6),np.round(expected_global_K_disp,6))) is True

# def element_internal_force(self,k_const):
#         """
#         Function for computing internal force vector for an element
        
#         Parameters
#         ----------
#         k_const : float64
#             parameter to avoid overflow for cracked elements.

#         Returns
#         -------
#         F_int_elem : Array of float64, size(num_elem_var_u)
#             internal force vector for an element.

#         """
#         num_dof_u = self.num_dof-1
        
#         # total number of displacements per element 
#         num_elem_var_u = self.num_node_elem*num_dof_u
        
#         # Initialize element internal force vector   
#         F_int_elem = np.zeros(num_elem_var_u)
        
#         i_tot_var = self.elem_node
#         elem_phi = self.Phi[i_tot_var-1]
        

#         for j in range(0,self.num_Gauss_2D):
#             gpos = self.Points[j]
            
#             # Calling shape function and its derivatives from shape function class
#             shape = shape_function(self.num_node_elem,gpos,self.elem_coord)
#             N = shape.get_shape_function()
#             dNdX = shape.get_shape_function_derivative()
#             # Call detereminant of Jacobian
#             det_Jacobian = shape.get_det_Jacobian()
            
#             # phase field order parameter
#             phi = np.matmul(N[0],elem_phi)
#             if phi > 1:
#                 phi = 1 
            
#             # Compute B matrix
#             B = Bmatrix(dNdX,self.num_node_elem)
#             Bmat = B.Bmatrix_disp()
            
#             # Compute internal force vector for an element
#             F_int_elem = F_int_elem + np.matmul(np.transpose(Bmat),self.stress[self.elem,j,:])*(((1-phi)**2)+k_const) \
#                 *self.Weights[j]*det_Jacobian
            
#         return F_int_elem

# def t2_otimes_t2(self,A,B):
#         """
        

#         Parameters
#         ----------
#         A : TYPE
#             DESCRIPTION.
#         B : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         C4 : TYPE
#             DESCRIPTION.

#         """
#         n = len(A)
#         C4 = np.zeros((n,n,n,n))
#         for i in range(n):
#             for j in range(n):
#                 for k in range(n):
#                     for l in range(n):
#                         C4[i,j,k,l] = A[i,j]*B[k,l]
        
        # return C4
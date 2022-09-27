# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from geometry import quadrature_coefficients
from material_routine import material_routine 
from element_staggered import element_staggered
from assembly_index import assembly
from boundary_condition import boundary_condition
from solve_stress_strain import solve_stress_strain
from input_abaqus import *
# from global_assembly import global_assembly
# from plots import *
import time

start_time = time.time()
# total number of variables in the solution
num_tot_var = num_node*num_dof

# the number of DOFs for displacements per node
num_dof_u = num_dof-1

# the number of DOFs for order parameters per node
num_dof_phi = num_dof-2

# total number of variables for displacements
num_tot_var_u = num_node*num_dof_u

# total number of variables for displacements
num_tot_var_phi = num_node*num_dof_phi

# total number of Gauss points used for integration in n-dimension
num_Gauss_2D = num_Gauss**num_dim

# Initialize stress component values at the integration points of all elements
stress = np.zeros((num_elem,num_Gauss_2D,num_stress))

# Initialize strain component values at the integration points of all elements 
strain = np.zeros((num_elem,num_Gauss_2D,num_stress))

strain_energy = np.zeros((num_elem,num_Gauss_2D))
# Initialize displacment values
disp = np.zeros(num_tot_var_u)

# Initialize phi (field parameter) values
phi = np.zeros(num_tot_var_phi)

# # Getting the nodes near crack
# crack = []
# for i in range(0,num_node):
#     if 0 <= nodes[i,0] and nodes[i,1] == 0:
#         crack.append(i)

# # assigning order parameter values as 1
# for i in range(0,len(crack)):
#     phi[crack[i]] = 1
    

# Call Guass quadrature points and weights using inbuilt function
Points,Weights = quadrature_coefficients(num_Gauss)

# Call Stiffness tensor from material routine
mat = material_routine()
if stressState == 1:    
    C = mat.planestress(Young,Poisson)
elif stressState == 2:
    C = mat.planestrain(Young,Poisson)

# total increment factor for displacement increments
tot_inc = 0

# Boundary points
bot = np.where(nodes[:,1] == min(nodes[:,1]))[0] # bottom edge nodes
left = np.where(nodes[:,0] == min(nodes[:,0]))[0] # left edge nodes
right = np.where(nodes[:,0] == max(nodes[:,0]))[0] # right edge nodes
top = np.where(nodes[:,1] == max(nodes[:,1]))[0] # top edge nodes

nodes_bc = np.r_[bot,top]
disp_bot = np.zeros((len(bot),num_dim))
disp_top = np.zeros((len(top),num_dim))

fixed_dof_bot = np.zeros((len(bot),num_dim),int) # fixed degrees of freedom for each node
fixed_dof_top = np.zeros((len(top),num_dim),int) # fixed degrees of freedom for each node
# disp_top[:,1] = disp_inc
fixed_dof_bot[:,0] = 1
fixed_dof_bot[:,1] = 1
fixed_dof_top[:,1] = 1

fixed_dof = np.r_[fixed_dof_bot,fixed_dof_top]
disp_bc = np.r_[disp_bot,disp_top]


# External force vector
F_ext = np.zeros(num_tot_var_u)

# Reaction force vector
R_ext = np.zeros(num_tot_var_u)

# Internal force vector
F_int = np.zeros(num_tot_var_u)

# disp_bc_new = np.zeros(num_tot_var_u)
# globel = global_assembly(num_elem,Points,Weights,C,disp,phi,stress,strain,num_tot_var_u,num_tot_var_phi,num_dof_u,num_dof_phi)
for step in range(0,num_step):
    step_time_start = time.time()
    tot_inc = tot_inc + disp_inc
    
    disp_top[:,1] = tot_inc
    disp_bc = np.r_[disp_bot,disp_top]
    
    # Initialization of global force vector and global stiffness matrix for  displacement
    global_force_disp = np.zeros(num_tot_var_u)
    global_K_disp = np.zeros((num_tot_var_u, num_tot_var_u))

    for elem in range(0,num_elem):
        elem_stiff = element_staggered(elem,Points,Weights,C,disp,phi,stress,strain,strain_energy)
        K_uu = elem_stiff.element_stiffness_displacement()
        assemble = assembly()
        
        index_u = assemble.assembly_index_u(elem,num_dof_u)
        
        X,Y = np.meshgrid(index_u,index_u,sparse=True)
        global_K_disp[X,Y] =  global_K_disp[X,Y] + K_uu
    
    global_force_disp = F_int - F_ext
    
    # Impose essential boundary conditions
    global_K_disp,global_force_disp = boundary_condition(num_dof_u,nodes_bc,fixed_dof,global_K_disp,global_force_disp,disp,disp_bc,tot_inc)
    max_iteration_reached = False            
    for iteration in range(max_iter):
        
        sp_global_K_disp = csc_matrix(global_K_disp)
        disp_solve = spsolve(sp_global_K_disp,global_force_disp)
        
        disp = disp_solve
        
        stress, strain = solve_stress_strain(num_elem,num_Gauss_2D,num_stress,num_node_elem,num_dof_u,elements,disp,Points,nodes,C)
        
        for elem in range(0,num_elem):
            elem_stiff = element_staggered(elem,Points,Weights,C,disp,phi,stress,strain,strain_energy)
            F_int_elem = elem_stiff.element_internal_force()
            assemble = assembly()
            
            index_u = assemble.assembly_index_u(elem,num_dof_u)
            
            F_int[index_u] = F_int[index_u]+F_int_elem
        
        # update global force and check convergence
        global_force_disp = F_int - F_ext
        tolerance = np.linalg.norm(global_force_disp)
        
        print("Solving for displacement: Iteration ---> {} --- tolerance = {}".format(iteration+1, tolerance))
        if (tolerance < max_tol):
            print("Displacement solution converged!")
            break 
        
        if (iteration == max_iter-1):
            print("Displacement solution has achieved its maximum number of iterations.!")
            print("Displacement solution failed to converge!")
            max_iteration_reached = True
            break 
    if(max_iteration_reached == True):
        break 
    # Initialization of global force vector and global stiffness matrix for phase field parameter
    global_force_phi = np.zeros(num_tot_var_phi)
    global_K_phi = np.zeros((num_tot_var_phi, num_tot_var_phi))
         
    for elem in range(num_elem):
        elem_stiff = element_staggered(elem,Points,Weights,C,disp,phi,stress,strain,strain_energy)
        K_phiphi,residual_phi,H_n = elem_stiff.element_stiffness_field_parameter()
        assemble = assembly()
        
        index_phi = assemble.assembly_index_phi(elem,num_dof_phi,num_tot_var_phi)
        
        X,Y = np.meshgrid(index_phi,index_phi,sparse=True)
        global_K_phi[X,Y] =  global_K_phi[X,Y] + K_phiphi
        # global_force_phi[index_phi] = global_force_phi[index_phi]+residual_phi

    for iteration in range(max_iter):
        sp_global_K_phi = csc_matrix(global_K_phi)
        phi_solve = spsolve(sp_global_K_phi,-global_force_phi)
        
        phi = phi_solve
        
        for i in range(num_tot_var_phi):
            if phi[i] > 1:
                phi[i] = 1 
            if phi[i] < 1:
                phi[i] = 0
        
        for elem in range(num_elem):
            elem_stiff = element_staggered(elem,Points,Weights,C,disp,phi,stress,strain,strain_energy)
            residual_elem_phi,H_n = elem_stiff.element_residual_field_parameter()
            assemble = assembly()
            
            index_phi = assemble.assembly_index_phi(elem,num_dof_phi,num_tot_var_phi)

            global_force_phi[index_phi] = global_force_phi[index_phi]+residual_elem_phi
        tolerance = np.linalg.norm(global_force_phi)
        print("Solving for phase field: Iteration ---> {} --- tolerance = {}".format(iteration+1, tolerance))
        if (tolerance < max_tol):
            print("Phase field solution converged!")
            break 
        if (iteration == max_iter-1):
            print("Phase field solution has achieved its maximum number of iterations.!")
            print("Phase field solution failed to converge!")
            max_iteration_reached = True
            break 
    if(max_iteration_reached == True):
        break 
    step_time_end = time.time()
    print('step:',step+1)
    print('step time:',step_time_end-step_time_start)
    
    
end_time = time.time()
print('total time:',end_time-start_time)

# plot_nodes_and_boundary(nodes)
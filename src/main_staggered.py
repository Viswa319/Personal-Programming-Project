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
from shape_function import shape_function
from Bmatrix import Bmatrix
import matplotlib.pyplot as plt
# from global_assembly import global_assembly
from plots import *
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

# Initialize strain energy 
strain_energy = np.zeros((num_elem,num_Gauss_2D))

# Initialize displacment values and displacement boundary conditions
disp = np.zeros(num_tot_var_u)
disp_bc = np.zeros(num_tot_var_u)

# Initialize phi (field parameter) values
phi = np.zeros(num_tot_var_phi)

# External force vector
F_ext = np.zeros(num_tot_var_u)

# Internal force vector
F_int = np.zeros(num_tot_var_u)

# total increment factor for displacement increments
tot_inc = 0

# Call Guass quadrature points and weights using inbuilt function
Points,Weights = quadrature_coefficients(num_Gauss)

# Call Stiffness tensor from material routine
mat = material_routine(Young,Poisson)
if stressState == 1: # Plane stress   
    C = mat.planestress()
elif stressState == 2: # Plane strain
    C = mat.planestrain()

# Boundary points for all edges
bot = np.where(nodes[:,1] == min(nodes[:,1]))[0] # bottom edge nodes
left = np.where(nodes[:,0] == min(nodes[:,0]))[0] # left edge nodes
right = np.where(nodes[:,0] == max(nodes[:,0]))[0] # right edge nodes
top = np.where(nodes[:,1] == max(nodes[:,1]))[0] # top edge nodes

# Getting the fixed degrees of freedom at all nodes
# If boundary condition is prescribed at certain node, then it is given value 1 or else 0
fixed_dof = np.zeros(num_tot_var_u)
fixed_dof[(top*2)+1] = 1
# fixed_dof[(left*2)] = 1
fixed_dof[(bot*2)+1] = 1
fixed_dof[bot*2] = 1

disp_plot = []
force_plot = []
# step = 0
# while tot_inc <= 0.01:
for step in range(num_step):
    # step = step+1
    print('step:',step+1)
    step_time_start = time.time() # for computing step time
    
    
    # if tot_inc <= 0.005:
    #     disp_inc = 1e-5
    # else:
    #     disp_inc = 1e-6
      
    # displacement load increment for every step
    tot_inc = tot_inc + disp_inc
    # prescribing essential boundary conditions 
    disp_bc[top*2+1] = tot_inc
    
    max_iteration_reached = False
    
    get_stress = solve_stress_strain(num_elem,num_Gauss_2D,num_stress,num_node_elem,num_dof_u,elements,disp,Points,nodes,C,strain_energy)
    strain_energy = get_stress.solve_strain_energy
    
    # Initialization of global force vector and global stiffness matrix for phase field parameter
    global_force_phi = np.zeros(num_tot_var_phi)
    global_K_phi = np.zeros((num_tot_var_phi, num_tot_var_phi))
         
    for elem in range(num_elem):
        elem_stiff = element_staggered(elem,Points,Weights,disp,phi,stress,strain,strain_energy,elements,nodes,num_dof,num_node_elem,num_Gauss_2D)
        K_phiphi = elem_stiff.element_stiffness_field_parameter(G_c,l_0)
            
        assemble = assembly()
        
        index_phi = assemble.assembly_index_phi(elem,num_dof_phi,num_tot_var_phi,num_node_elem,elements)
        
        X,Y = np.meshgrid(index_phi,index_phi,sparse=True)
        global_K_phi[X,Y] =  global_K_phi[X,Y] + K_phiphi
        del X,Y
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
            elem_stiff = element_staggered(elem,Points,Weights,disp,phi,stress,strain,strain_energy,elements,nodes,num_dof,num_node_elem,num_Gauss_2D)
            residual_elem_phi = elem_stiff.element_residual_field_parameter(G_c,l_0)
            assemble = assembly()
            
            index_phi = assemble.assembly_index_phi(elem,num_dof_phi,num_tot_var_phi,num_node_elem,elements)

            global_force_phi[index_phi] = global_force_phi[index_phi]+residual_elem_phi
        tolerance = np.linalg.norm(global_force_phi)
        print("Solving for phase field: Iteration ---> {} --- tolerance = {}".format(iteration+1, tolerance))
        if (tolerance < max_tol):
            print("Phase field solution converged!")
            break 

        if (iteration == max_iter-1):
            print("Phase field solution has achieved its maximum number of iterations and failed to converge.!")
            max_iteration_reached = True
            break 
    if(max_iteration_reached == True):
        break
    
    # Initialization of global force vector and global stiffness matrix for  displacement
    global_force_disp = np.zeros(num_tot_var_u)
    global_K_disp = np.zeros((num_tot_var_u, num_tot_var_u))

    for elem in range(0,num_elem):
        elem_stiff = element_staggered(elem,Points,Weights,disp,phi,stress,strain,strain_energy,elements,nodes,num_dof,num_node_elem,num_Gauss_2D)
        K_uu = elem_stiff.element_stiffness_displacement(C,k_const)
        assemble = assembly()
        
        index_u = assemble.assembly_index_u(elem,num_dof_u,num_node_elem,elements)
        
        X,Y = np.meshgrid(index_u,index_u,sparse=True)
        global_K_disp[X,Y] =  global_K_disp[X,Y] + K_uu
        del X,Y
    global_force_disp = F_int - F_ext
    
    # Impose essential boundary conditions
    global_K_disp,global_force_disp = boundary_condition(num_dof_u,fixed_dof,global_K_disp,global_force_disp,disp,disp_bc)
    
    for iteration in range(max_iter):
        
        sp_global_K_disp = csc_matrix(global_K_disp)
        disp_reduced = spsolve(sp_global_K_disp,-global_force_disp)
        
        disp = disp_reduced
        
        get_stress = solve_stress_strain(num_elem,num_Gauss_2D,num_stress,num_node_elem,num_dof_u,elements,disp,Points,nodes,C,strain_energy)
        strain = get_stress.solve_strain
        stress = get_stress.solve_stress
        
        for elem in range(0,num_elem):
            elem_stiff = element_staggered(elem,Points,Weights,disp,phi,stress,strain,strain_energy,elements,nodes,num_dof,num_node_elem,num_Gauss_2D)
            F_int_elem = elem_stiff.element_internal_force(k_const)

            assemble = assembly()

            index_u = assemble.assembly_index_u(elem,num_dof_u,num_node_elem,elements)
            
            F_int[index_u] = F_int[index_u]+F_int_elem
        
        tolerance = np.linalg.norm(global_force_disp)
        # update global force and check convergence
        # global_force_disp = F_int - F_ext
        
        print("Solving for displacement: Iteration ---> {} --- norm_1 = {} and norm_2 = {}".format(iteration+1, np.linalg.norm(global_force_disp),0.005*np.linalg.norm(F_int)))
        # if (tolerance < max_tol):
        #     print("Displacement solution converged!")
        #     break 
        
        if (np.linalg.norm(global_force_disp) < 0.005*np.linalg.norm(F_int)):
            print("Displacement solution converged!")
            break 
        if (iteration == max_iter-1):
            print("Displacement solution has achieved its maximum number of iterations and failed to converge.!")
            max_iteration_reached = True
            break 
    if(max_iteration_reached == True):
        break 
    
    step_time_end = time.time()
    print('step time:',step_time_end-step_time_start)
    
    disp_plot.append(tot_inc)
    force_plot.append(sum(F_int[(bot*2)+1]))
end_time = time.time()
print('total time:',end_time-start_time)
plt.plot(disp_plot,force_plot)
plot_nodes_and_boundary(nodes)
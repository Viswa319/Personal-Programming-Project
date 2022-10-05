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
from element_stiffness import element_stiffness
from assembly_index import assembly
from boundary_condition import boundary_condition
from solve_stress_strain import solve_stress_strain
from input_parameters import *
from residual import residual
import time
from vtk_generator import vtk_generator

start_time = time.time()
# total number of variables in the solution
num_tot_var = num_node*num_dof

# the number of DOFs for displacements per node
num_dof_u = num_dof-1

# the number of DOFs for order parameters per node
num_dof_phi = num_dof-2

# total number of variables for displacements
num_tot_var_u = num_node*num_dof_u

# total number of Gauss points used for integration in n-dimension
num_Gauss_2D = num_Gauss**num_dim

# Initialize stress component values at the integration points of all elements
stress = np.zeros((num_elem,num_Gauss_2D,num_stress))

# Initialize strain component values at the integration points of all elements 
strain = np.zeros((num_elem,num_Gauss_2D,num_stress))

# Initialize strain energy 
strain_energy = np.zeros((num_elem,num_Gauss_2D))

# Initialize displacment values
disp = np.zeros(num_tot_var)

# Getting the nodes near crack
crack = []
for i in range(0,num_node):
    if 0 <= nodes[i,0] <= 3.5 and nodes[i,1] == 9.9:
        crack.append(i)
    if 0 <= nodes[i,0] <= 3.5 and nodes[i,1] == 10:
        crack.append(i)

# assigning order parameter values as 0.99 (approximately 1)
for i in range(0,len(crack)):
    disp[num_tot_var_u+crack[i]] = 0.99
    

# Call Guass quadrature points and weights using inbuilt function
Points,Weights = quadrature_coefficients(num_Gauss)

# Call Stiffness tensor from material routine
mat = material_routine(Young,Poisson)
if stressState == 1: # Plane stress   
    C = mat.planestress()
elif stressState == 2: # Plane strain
    C = mat.planestrain()

# total increment factor for displacement increments
tot_inc = 0

for step in range(0,num_step):
    step_time_start = time.time()
    tot_inc = tot_inc + disp_inc
    disp_old = disp
    
    # Initialization of global force vector and global stiffness matrix
    global_force = np.zeros(num_tot_var)
    global_K = np.zeros((num_tot_var, num_tot_var))
    
    start_assembly_time = time.time()    
    for elem in range(0,num_elem):
        
        elem_stiffness = element_stiffness()
        K_uu,K_uphi,K_phiu,K_phiphi = elem_stiffness.element_stiffness_monolithic(elem,Points,Weights,C,disp,disp_old,stress,strain)
        assemble = assembly()
        
        index_u = assemble.assembly_index_u(elem,num_dof_u,num_node_elem,elements)
        index_phi = assemble.assembly_index_phi(elem,num_dof_phi,num_tot_var_u,num_node_elem,elements)
        
        X,Y = np.meshgrid(index_u,index_u,sparse=True)
        global_K[X,Y] =  global_K[X,Y] + K_uu
        
        X,Y = np.meshgrid(index_u,index_phi,sparse=True)
        global_K[X,Y] =  global_K[X,Y] + np.transpose(K_uphi)
        
        X,Y = np.meshgrid(index_phi,index_u,sparse=True)
        global_K[X,Y] =  global_K[X,Y] + np.transpose(K_phiu)
        
        X,Y = np.meshgrid(index_phi,index_phi,sparse=True)
        global_K[X,Y] =  global_K[X,Y] + K_phiphi
    end_assembly_time = time.time()
    
    for iteration in range(0,max_iter):
        global_K,global_force,external_force = boundary_condition(num_dof_u,nodes_bc,fixed_dof,global_K,global_force,disp,disp_bc,tot_inc)
        start_solve_time = time.time()
        
        sp_global_K = csc_matrix(global_K)
        disp_solve = spsolve(sp_global_K,global_force)
        
        end_solve_time = time.time()
        disp = disp + disp_solve
        
        for i in range(num_tot_var_u,num_tot_var):
            if disp[i] > 0.999:
                disp[i] = 0.999
            if disp[i] < 0:
                disp[i] = 0
        
        get_stress = solve_stress_strain(num_elem,num_Gauss_2D,num_stress,num_node_elem,num_dof_u,elements,disp,Points,nodes,C,strain_energy)
        strain = get_stress.solve_strain
        stress = get_stress.solve_stress
        
        if np.linalg.norm(global_force) <= tol:
            break
        global_force = residual(num_node,elem,num_node_elem,num_dim,num_dof,elements,nodes,num_Gauss,Points,Weights,C,disp,disp_old,stress,strain,global_force)
    step_time_end = time.time()
    print('step:',step+1)
    print('step time:',step_time_end-step_time_start)
    print('assembly time:',end_assembly_time-start_assembly_time)
    print('solve time:',end_solve_time-start_solve_time)
    
    if np.mod(step,num_print-1) == 0:
        deflection = np.zeros((num_node,num_dim))
        order_parameter = np.zeros(num_node)
        for i in range(0,num_node):
            for j in range(0,num_dof_u):
                itotv = i*num_dof+j
                deflection[i,j] = nodes[i,j]+10*disp[itotv]
                jtotv = num_tot_var_u+i
                order_parameter[i] = disp[jtotv]
        file_name = 'time_{}.vtk'.format(step)
        vtk_generator(file_name,deflection,order_parameter)
    
end_time = time.time()
print('total time:',end_time-start_time)

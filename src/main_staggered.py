"""
Implementation of phase-field model of ductile fracture
*******************************************************

Phase-field model for elastic and elastic-plastic case is implemented using staggered scheme.

The whole program is divided into pre-processing, processing and post.processing stages.\n

1. **Pre-processing:** The pre-processing stage invloves the following steps, \n

    - Defining geometry, generating nodes and elements \n
    - Calling input data \n
    - Initialize all arrays \n
    - Generating boundary points \n

2. **Processing:** The processing stage involves the following steps, \n
    - Calculating Gauss quadrature points and weights \n
    - Calculating stiffness tensor (for elastic case) \n
    - Loop for all time steps \n
        - Compute strain energy and check with the previous value and update accordingly \n
        - Compute stiffness element matrices and residual vectors for every element for phase-field order parameter \n
        - Assemble element matrices and vectors into global stiffness matrix and residual vector \n
        - Newton-Raphson iteration loop for phase field order parameter\n
            - Solve the matrix equation for order parameter values \n
            - Compute element residual vector and assemble into global residual vector \n
            - Check for convergence \n
        - Newton-Raphson iteration loop for displacement \n
            - Compute stiffness element matrices and residual vectors for every element for phase-field order parameter \n
            - Assemble element matrices and vectors into global stiffness matrix and residual vector \n
            - Apply essential boundary conditions to global stiffness matrix and residual vector \n
            - Solve the matrix equation for displacement values \n
            - Update displacement values by including essential boundary conditions \n
            - Compute stress and strain values \n
            - Check for convergence
3. **Post-processing:** The post-processing stage invloves the following steps, \n
    - Generating phase field order parameter distribution contour plot \n
    - Generating load vs displacemnt curve \n
    - Extracting displacement data for generating contour plots

"""
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
from solve_stress_strain import solve_stress_strain
from input_abaqus import input_parameters
from plots import plot_load_displacement, plot_field_parameter, plot_deformation_with_nodes
import time
import json
import matplotlib as mpl
mpl.rc('figure', max_open_warning = 0)
def main():
    # asking user about the load type tensile or shear
    load_type = int(input("Choose a load type, 0 for tensile and 1 for shear:"))
    
    # asking user about the problem type elastic or elastic-plastic
    problem = int(input("Choose a problem type, 0 for elastic case and 1 for elastic-plastic case:"))
    # if problem == 1:
    #     # Asking user whether to perform elastic-plastic brittle or ductile case
    #     problem = int(input("Give input as 1 for elastic-plastic brittle and 2 for elastic-plastic ductile case:"))
    
    start_time = time.time()  # for computing time taken to run whole program
    
    inputs = input_parameters(load_type, problem)
    num_dim, num_node_elem, nodes, num_node, elements, num_elem, num_dof, num_Gauss, num_stress = inputs.geometry_parameters()
    if problem == 0:
        k_const, G_c, l_0, Young, Poisson, stressState = inputs.material_parameters_elastic()
    elif problem == 1 or problem == 2:
        k_const, G_c, l_0, stressState, shear, bulk, sigma_y, hardening = inputs.material_parameters_elastic_plastic()
    
    num_step, max_iter, max_tol,  disp_inc = inputs.time_integration_parameters()
    
    # the number of DOFs for displacements per node
    num_dof_u = num_dof - 1
    
    # the number of DOFs for order parameters per node
    num_dof_phi = num_dof - 2
    
    # total number of variables for displacements
    num_tot_var_u = num_node * num_dof_u
    
    # total number of variables for order parameters
    num_tot_var_phi = num_node * num_dof_phi
    
    # total number of Gauss points used for integration in n-dimension
    num_Gauss_2D = num_Gauss**num_dim
    
    # Initialize stress component values at the integration points for all elements
    stress = np.zeros((num_elem, num_Gauss_2D, num_stress))
    
    # Initialize strain component values at the integration points for all elements
    strain = np.zeros((num_elem, num_Gauss_2D, num_stress))
    
    # Initialize strain energy at the integration points for all elements
    strain_energy = np.zeros((num_elem, num_Gauss_2D))
    
    if problem == 1:
        # Initialize plastic strain component values at the integration points for all elements
        # Initializes only if the problem is elastic-plastic
        strain_plas = np.zeros((num_elem, num_Gauss_2D, num_stress))
        
        # Initialize scalar hardening variable
        alpha = np.zeros(num_elem)
    
    
    # Initialize Eéxternal force vector
    F_ext = np.zeros(num_tot_var_u)
    
    # Initialize displacment and displacement boundary conditions vector with zeros
    disp = np.zeros(num_tot_var_u)
    disp_bc = np.zeros(num_tot_var_u)
    
    # Initialize phi (field parameter) vector with zeros
    phi = np.zeros(num_tot_var_phi)
    
    # Boundary points for all edges
    bot = np.where(nodes[:, 1] == min(nodes[:, 1]))[0]  # bottom edge nodes
    left = np.where(nodes[:, 0] == min(nodes[:, 0]))[0]  # left edge nodes
    right = np.where(nodes[:, 0] == max(nodes[:, 0]))[0]  # right edge nodes
    top = np.where(nodes[:, 1] == max(nodes[:, 1]))[0]  # top edge nodes
    
    # Getting the fixed degrees of freedom at all nodes
    # if boundary condition is prescribed at certain node,
    # then it is given as 1 or else 0
    
    if load_type == 0:
        # Tensile problem essential boundary conditions
        fixed_dof = np.zeros(num_tot_var_u)
        fixed_dof[(top * 2) + 1] = 1  # prescribed for top nodes in y-direction
        fixed_dof[(top * 2)] = 1  # prescribed for top nodes in x-direction
        fixed_dof[(bot * 2) + 1] = 1  # prescribed for bottom nodes in y-direction
        fixed_dof[bot * 2] = 1  # prescribed for bottom nodes in x-direction
        
    
    elif load_type == 1:
        # Shear problem essential boundary conditions
        fixed_dof = np.zeros(num_tot_var_u)
        fixed_dof[(top * 2) + 1] = 1  # prescribed for top nodes in y-direction
        fixed_dof[(top * 2)] = 1  # prescribed for top nodes in x-direction
        fixed_dof[(left * 2) + 1] = 1  # prescribed for left nodes in y-direction
        fixed_dof[(right * 2) + 1] = 1  # prescribed for right nodes in y-direction
        fixed_dof[(bot * 2) + 1] = 1  # prescribed for bottom nodes in y-direction
        fixed_dof[bot * 2] = 1  # prescribed for bottom nodes in x-direction
    
    # Getting arrays where boundary points are fixed and where not fixed
    Fixed = np.where(fixed_dof == 1)[0]
    Not_fixed = np.where(fixed_dof != 1)[0]
    
    # Call Guass quadrature points and weights from geometry
    Points, Weights = quadrature_coefficients(num_Gauss)
    
    # # Call Stiffness tensor from material routine
    mat = material_routine(problem,load_type)
    C = mat.material_elasticity()
    
    # total increment factor for displacement increments
    tot_inc = 0
    
    # Create an empty list for storing displacement increment and load values
    # which are used for generating plots
    disp_plot = []
    force_plot = []
    
    # Initialize an empty dictionary for storing displacement and phi values
    datadict = {}
    datadict_2 = {}
    
    for step in range(num_step):
        print('step:', step + 1)
        step_time_start = time.time()  # for computing step time
    
        # displacement load increment for every step
        tot_inc = tot_inc + disp_inc
    
        # prescribing essential boundary conditions
        if load_type == 0:
            disp_bc[top * 2 + 1] = tot_inc  # prescribed for top nodes in y-direction
        if load_type == 1:
            disp_bc[top * 2] = tot_inc  # prescribed for top nodes in x-direction
         
        max_iteration_reached = False
    
        # strain energy is computed and updated accordingly in solve_stress_strain class
        if problem == 0:
            get_stress = solve_stress_strain(num_Gauss_2D, num_stress, num_node_elem, num_dof_u, elements, disp, Points, nodes, C, strain_energy, problem)
        elif problem == 1:
            get_stress = solve_stress_strain(num_Gauss_2D, num_stress, num_node_elem, num_dof_u, elements, disp, Points, nodes, C, strain_energy, problem, strain_plas, alpha, hardening, sigma_y)
        strain_energy = get_stress.solve_strain_energy
    
        # Initialization of global force vector and global stiffness matrix for phase field parameter
        global_force_phi = np.zeros(num_tot_var_phi)
        global_K_phi = np.zeros((num_tot_var_phi, num_tot_var_phi))
        global_force_phi_tol = np.zeros(num_tot_var_phi)
        
        # Calling element routine and assemble all the element stiffness matrices and element residual vectors
        # to a global element stiffness matrix and global residual vector respectively
        for elem in range(num_elem):
            elem_stiff = element_staggered(elem, Points, Weights, disp, phi, stress, strain_energy, elements, nodes, num_dof, num_node_elem, num_Gauss_2D)
            K_phiphi, residual_elem_phi = elem_stiff.element_stiffness_field_parameter(G_c, l_0)
            assemble = assembly()
    
            index_phi = assemble.assembly_index_phi(elem, num_dof_phi, num_node_elem, elements)
    
            X, Y = np.meshgrid(index_phi, index_phi, sparse=True)
            global_K_phi[X, Y] = global_K_phi[X, Y] + K_phiphi
            global_force_phi[index_phi] = global_force_phi[index_phi] + residual_elem_phi
            del X, Y
        
        # Newton Raphson iteration for phase field
        for iteration in range(max_iter):
            
            # converting global stiffness matrix to sparse matrix
            sp_global_K_phi = csc_matrix(global_K_phi)
            
            # solving for phi (order parameter)
            delta_phi = spsolve(sp_global_K_phi, - global_force_phi)
            # Update phi (fractue field parameter)
            phi = phi + delta_phi
            
            # Compute element residual vector with updated phi and assmeble all to a global residual vector
            for elem in range(num_elem):
                elem_stiff = element_staggered(elem, Points, Weights, disp, phi, stress, strain_energy, elements, nodes, num_dof, num_node_elem, num_Gauss_2D)
                residual_elem_phi = elem_stiff.element_residual_field_parameter(G_c, l_0)
                assemble = assembly()
    
                index_phi = assemble.assembly_index_phi(elem, num_dof_phi, num_node_elem, elements)
    
                global_force_phi_tol[index_phi] = global_force_phi_tol[index_phi] + residual_elem_phi
            
            # tolerance for convergence criteria
            tolerance = np.linalg.norm(global_force_phi_tol)
            print(f'Solving for phase field: Iteration ---> {iteration + 1} --- norm = {np.linalg.norm(global_force_phi_tol)}')
    
            # Checking for convergence
            if (tolerance < max_tol):
                print('Phase field solution converged!!!')
                break
    
            if (iteration == max_iter - 1):
                print('Phase field solution has achieved its maximum number of iterations and failed to converge.!!!')
                max_iteration_reached = True
                break
    
        if(max_iteration_reached):
            # If maximum number of iterations reached break the solution
            break
    
        # Newton Raphson iteration for displacement
        for iteration in range(max_iter):
            # Initialization of global force vector and global stiffness matrix for  displacement
            global_K_disp = np.zeros((num_tot_var_u, num_tot_var_u))
    
            # initialize global internal force vector
            F_int = np.zeros(num_tot_var_u)
    
            for elem in range(0, num_elem):
                elem_stiff = element_staggered(elem, Points, Weights, disp, phi, stress, strain_energy, elements, nodes, num_dof, num_node_elem, num_Gauss_2D)
                
                if problem == 0:
                    K_uu, F_int_elem = elem_stiff.element_stiffness_displacement(C, k_const)
                if problem == 1:
                    K_uu, F_int_elem, stress, strain_plas, alpha_cur = elem_stiff.element_stiffness_displacement_plasticity(k_const, strain, strain_plas, alpha[elem], problem)
                    alpha[elem] = alpha_cur
                assemble = assembly()
                index_u = assemble.assembly_index_u(elem, num_dof_u, num_node_elem, elements)
    
                X, Y = np.meshgrid(index_u, index_u, sparse=True)
                global_K_disp[X, Y] = global_K_disp[X, Y] + K_uu
                F_int[index_u] = F_int[index_u] + F_int_elem
                del X, Y
                
            
            global_force_disp = F_int - F_ext
            # Applying essential boundary conditions for global stiffness matrix and global residual vector,
            # by reducing the matrix and vector respectively
            global_K_disp_red = np.copy(global_K_disp)
            global_K_disp_red = np.delete(global_K_disp_red, Fixed, axis=0)
            global_K_disp_red = np.delete(global_K_disp_red, Fixed, axis=1)
            global_force_disp_red = np.copy(global_force_disp)
            global_force_disp_red = np.delete(global_force_disp_red, Fixed, axis=0)
    
            # converting global stiffness matrix to sparse matrix
            sp_global_K_disp = csc_matrix(global_K_disp_red)
    
            # solving for displacement
            delta_disp = spsolve(sp_global_K_disp, -global_force_disp_red)
            
            # Update displacement with boundary conditions
            disp[Not_fixed] = disp[Not_fixed] + delta_disp
            disp[Fixed] = disp_bc[Fixed]
    
            # stress and strain are computed using solve_stress_strain class
            if problem == 0:
                get_stress = solve_stress_strain(num_Gauss_2D, num_stress, num_node_elem, num_dof_u, elements, disp, Points, nodes, C, strain_energy, problem)
                strain = get_stress.solve_strain
                stress = get_stress.solve_stress
            elif problem == 1 or problem == 2:
                get_stress = solve_stress_strain(num_Gauss_2D, num_stress, num_node_elem, num_dof_u, elements, disp, Points, nodes, C, strain_energy, problem, strain_plas, alpha, hardening, sigma_y)
                strain = get_stress.solve_strain
    
            # print("Solving for displacement: Iteration ---> {} --- norm_1 = {} and norm_2 = {}".format(iteration + 1, np.linalg.norm(delta_disp, np.inf), 0.005 * np.linalg.norm(disp[Not_fixed], np.inf)))
            print("Solving for displacement: Iteration ---> {} --- norm_1 = {} and norm_2 = {}".format(iteration + 1, np.linalg.norm(global_force_disp_red), 0.005 * np.linalg.norm(F_int, np.inf)))
            
            # Checking for convergence
            if (np.linalg.norm(global_force_disp_red) < 0.005*max(np.linalg.norm(F_int,np.inf),10**-8)):
            # if np.linalg.norm(delta_disp, np.inf) < 0.005 * np.linalg.norm(disp[Not_fixed], np.inf):
                print("Displacement solution converged!!!")
                break
    
            if (iteration == max_iter - 1):
                print("Displacement solution has achieved its maximum number of iterations and failed to converge.!!!")
                max_iteration_reached = True
                break
    
        if (max_iteration_reached):
            # If maximum number of iterations reached break the solution
            break
    
        step_time_end = time.time()
        print('step time:', step_time_end - step_time_start)
    
        # storing displacement increment and load values which are used for generating plots
        disp_plot.append(tot_inc)
        if load_type == 0:
            force_plot.append(-sum(F_int[(bot * 2 + 1)]))
        if load_type == 1:
            force_plot.append(-sum(F_int[(bot * 2)]))
        # if (step + 1) % 1000 == 0:
        #     plot_field_parameter(nodes, phi, tot_inc)
        #     datadict_2[np.round(tot_inc,4)] = phi.tolist()
        #     datadict[np.round(tot_inc,4)] = disp.tolist()
    end_time = time.time()
    print('total time:', end_time - start_time)
    # with open("shear_disp.json", "w") as opf:    
    #     json.dump(datadict, opf, indent=4)
    # with open("shear_phi.json", "w") as opf:    
    #     json.dump(datadict_2, opf, indent=4)    
    plot_load_displacement(disp_plot, force_plot)
    X_Disp = disp[0:num_tot_var_u:2]
    Y_Disp = disp[1:num_tot_var_u:2]
    plot_deformation_with_nodes(nodes,X_Disp,Y_Disp)
if __name__ == "__main__":
    main()
# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np

class input_parameters():
    def __init__(self):
        pass
    
    def geometry_parameters(self):
        # dimension of a problem
        num_dim = 2
        
        # number of nodes per element
        num_node_elem = 4
        
        # with open('0_008.inp', "r") as f:
        
        #     lines = f.readlines()
        #     # number of nodes
        #     for i in range(len(lines)):
        #         if lines[i] == "*Nset, nset=Set-1, generate\n":
        #             num_node = int((lines[i + 1].split(',')[1]))
                
        #         if lines[i] == "*Node\n":
        #             start_node = i + 1
                
        #         # if lines[i] == "*ELEMENT, TYPE=U1, ELSET=SOLID\n":
        #         if lines[i] == "*Element, type=S4R\n":
        #             start_elem = i + 1
                
        #         if lines[i] == "*Elset, elset=Set-1, generate\n":
        #             num_elem = int((lines[i + 1].split(',')[1]))
        #             break
        
        #     # extracting nodes from input file
        #     nodes = np.zeros((num_node, num_dim), float)
        #     for i in range(start_node, start_node + num_node):
        #         nodes[i - start_node, 0] = (lines[i].split(','))[1]
        #         nodes[i - start_node, 1] = (lines[i].split(','))[2]
        
        
        #     # extracting elements and material type of each element from input file
        #     elements = np.zeros((num_elem, num_node_elem), int)
        #     for i in range(start_elem, start_elem + num_elem):
        #         elements[i - start_elem, 0:num_node_elem] = (lines[i].split(','))[1:num_node_elem + 1]
        
        #     del start_elem, start_node
        
        nodes = np.array([[0,0],[1,0],[1,1],[0,1]])
        num_node = len(nodes)
        elements = np.array([[1,2,3,4]])
        num_elem = len(elements)

        
        # total number of degrees of freedom ----> 2 displacement and 1 phase field
        num_dof = 3
        
        # number of Gauss points for integration
        num_Gauss = 2
        
        # number of independent stress components
        num_stress = 3
        
        problem = 'Elastic'
        return num_dim, num_node_elem, nodes,num_node,elements,num_elem, num_dof, num_Gauss, num_stress, problem
    
    def material_parameters_elastic(self):
        # Material specific parameters
        
        # parameter to avoid overflow for cracked elements
        k_const = 1e-7
        
        # critical energy release for unstable crack or damage
        G_c = 2.7  # MPa mm
        
        # length parameter which controls the spread of damage
        l_0 = 0.04  # mm
        
        # Young's modulus
        Young = 210000  # Mpa
        
        # Poisson's ratio
        Poisson = 0.3
        
        # if stressState == 1 plane stress; if stressState = 2 plane strain
        stressState = 2
        
        return k_const,G_c,l_0,Young,Poisson,stressState
        
    def material_parameters_elastic_plastic(self):
        # Material specific parameters
        
        # parameter to avoid overflow for cracked elements
        k_const = 1e-7
        
        # critical energy release for unstable crack or damage
        G_c = 2.7  # MPa mm
        
        # length parameter which controls the spread of damage
        l_0 = 0.04  # mm
        
        # Shear modulus
        shear = 27280  # Mpa
        
        # Bulk modulus
        bulk = 71600  # Mpa
        
        # Yield stress
        sigma_y = 345  # Mpa
        
        # Hardening modulus 
        hardening = 250  # Mpa
        
        # if stressState == 1 plane stress; if stressState = 2 plane strain
        stressState = 2
        
        return k_const, G_c, l_0, stressState, shear, bulk, sigma_y, hardening

    def time_integration_parameters(self):
        
        # Inputs for time integration parameters
        
        # number of time steps
        num_step = 1000
        
        # maximum number of Newton-Raphson iterations
        max_iter = 5
        
        # tolerance for iterative solution
        max_tol = 1e-4
        
        # time increment for numerical integration
        delta_time = 1
        
        # displacmenet increment per time steps
        disp_inc = 1e-5  # mm
        
        return num_step, max_iter, max_tol, delta_time, disp_inc
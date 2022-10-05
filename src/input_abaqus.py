# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
with open('Job-1.inp', "r") as f:
    
    lines = f.readlines()
    start_node = 17
    for i in range(start_node,len(lines)):
        if lines[i] == "*************************************************\n":
            end_node = i
            break
    # dimension of a problem
    num_dim = int((lines[end_node+1].split(',')[4])[-1])
    
    # number of nodes
    for i in range(start_node,len(lines)):
        if lines[i] == "*Nset, nset=Set-1, generate\n":
            num_node = int((lines[i+1].split(',')[1]))
            break
    # # number of nodes
    # num_node = end_node - start_node
    
    # number of elements
    for i in range(start_node,len(lines)):
        if lines[i] == "*Elset, elset=Set-1, generate\n":
            num_elem = int((lines[i+1].split(',')[1]))
            break
    # number of material properties
    num_props = int((lines[end_node+1].split(',')[3])[-1])
    
    # number of nodes per element
    num_node_elem = int((lines[end_node+1].split(',')[1])[-1])
    
    # extracting nodes from input file
    nodes = np.zeros((num_node,num_dim),float)
    for i in range(start_node,start_node+num_node):
        nodes[i-start_node,0] = (lines[i].split(','))[1]
        nodes[i-start_node,1] = (lines[i].split(','))[2]
    
    # number of material properties
    num_props = int((lines[end_node+1].split(',')[3])[-1])
    
    for i in range(end_node+1,len(lines)):
        if lines[i] == "*************************************************\n":
            start_elem = i+2
            break
    
    for i in range(start_elem+1,len(lines)):
        if lines[i] == "*************************************************\n":
            end_elem = i
            break
    
    # # number of elements
    # num_elem = end_elem-start_elem
    
    # extracting elements and material type of each element from input file 
    elements = np.zeros((num_elem,num_node_elem),int)
    for i in range(start_elem,start_elem+num_elem):
        elements[i-start_elem,0:num_node_elem] = (lines[i].split(','))[1:num_node_elem+1]
    
    del start_elem,end_elem,start_node,end_node
    
# nodes = np.array([[0,0],[1,0],[1,1],[0,1]])

# num_node = len(nodes)

# elements = np.array([[1,2,3,4]])
# num_elem = len(elements)

# num_node_elem = 4

# num_dim = 2

# if stressState == 1 plane stress; if stressState = 2 plane strain
stressState = 2
      
# total number of degrees of freedom ----> 2 displacement and 1 phase field
num_dof = 3
        
# number of Gauss points for integration
num_Gauss = 2

# number of independent stress components
num_stress = 3
        
# number of materials
num_mat = 1
    
# Material specific parameters

# parameter to avoid overflow for cracked elements
k_const = 1e-7

# critical energy release for unstable crack or damage
G_c = 2.7 # MPa mm

# length parameter which controls the spread of damage
l_0 = 0.04 # mm

# Young's modulus 
Young = 210000 # Mpa

# Poisson's ratio
Poisson = 0.3 

# Inputs for time integration parameters

# number of time steps
num_step = 10000

# print frequency to output the results to file
num_print = 25

# time increment for numerical integration
delta_time = 1.0

# maximum number of Newton-Raphson iterations
max_iter = 5

# tolerance for iterative solution
max_tol = 5e-3

# displacmenet increment per time steps
disp_inc = 1e-6 # mm
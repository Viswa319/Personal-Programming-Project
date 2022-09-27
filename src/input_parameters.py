# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
with open('fract_1ca.inp', "r") as f:
    lines = f.readlines()
        
    # number of nodes
    num_node = int(lines[0].split()[0])
    
    # number of elements
    num_elem = int(lines[0].split()[1])
    
    # number of fixed nodes
    num_fixnodes = int(lines[0].split()[2])
    
    # if stressState == 1 plane stress; if stressState = 2 plane strain
    stressState = int(lines[0].split()[3])
    
    # number of nodes per element
    num_node_elem = int(lines[0].split()[4])
    
    # total number of degrees of freedom
    num_dof = int(lines[0].split()[5])
        
    # dimension of a problem
    num_dim = int(lines[0].split()[6])
    
    # number of Gauss points for integration
    num_Gauss = int(lines[0].split()[7])

    # number of independent stress components
    num_stress = int(lines[0].split()[8])
        
    # number of materials
    num_mat = int(lines[0].split()[9])
        
    # number of material properties
    num_props = int(lines[0].split()[10]) 
        
    # extracting elements and material type of each element from input file 
    elements = np.zeros((num_elem,num_node_elem),int)
    mat_type = np.zeros(num_elem)
    for i in range(1,num_elem+1):
        elements[i-1,0:num_node_elem] = (lines[i].split())[1:num_node_elem+1]
        mat_type[i-1] = (lines[i].split())[num_node_elem+1]
        
    # extracting nodes from input file
    nodes = np.zeros((num_node,num_dim),float)
    for i in range(1+num_elem,1+num_elem+num_node):
        nodes[i-1-num_elem,0] = float((lines[i].split())[1])
        nodes[i-1-num_elem,1] = float((lines[i].split())[2])
          
    # extracting data related to boundary conditions from input file
    nodes_bc = np.zeros(num_fixnodes,int) # nodes where boundary conbditions are applied
    fixed_dof = np.zeros((num_fixnodes,num_dim),int) # fixed degrees of freedom for each node
    disp_bc = np.zeros((num_fixnodes,num_dim),float) # displacement applied at boundary conditions
    for i in range(1+num_elem+num_node,1+num_elem+num_node+num_fixnodes):
        nodes_bc[i-1-num_elem-num_node] = (lines[i].split())[0]
        fixed_dof[i-1-num_elem-num_node,0:num_dim] = (lines[i].split())[1:num_dim+1]
        disp_bc[i-1-num_elem-num_node,0:num_dim] = (lines[i].split())[3:num_dim+3]
        
    # extracting material properties from input file
    props = np.zeros(num_props,float)
    props[0:num_props] = (lines[1+num_elem+num_node+num_fixnodes].split())[1:num_props+1]
    
    # point load
    num_point_load = int((lines[1+num_elem+num_node+num_fixnodes+1].split())[0])
    
    # number of edged having distributed loads
    num_edge_load = int((lines[1+num_elem+num_node+num_fixnodes+1].split())[1])
    
# Material specific parameters

# parameter to avoid overflow for cracked elements
k_const = 1e-6

# critical energy release for unstable crack or damage
G_c = 0.001

# length parameter which controls the spread of damage
l_0 = 0.125

# penalty parameter which is taken as 2
n_const = 2

# parameter  which controls the magnitude of the penalty term
neta = (20*10**5)*2

# Young's modulus
Young = props[0]

# Poisson's ratio
Poisson = props[1]

# Inputs for time integration parameters

# number of time steps
num_step = 4000

# print frequency to output the results to file
num_print = 25

# time increment for numerical integration
delta_time = 1.0

# maximum number of Newton-Raphson iterations
max_iter = 5

# tolerance for iterative solution
tol = 5e-3

# displacmenet increment per time steps
disp_inc = 0.0005
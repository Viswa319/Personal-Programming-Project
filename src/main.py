# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
import numpy.polynomial.legendre as ptwt
from geometry import quadrature_coefficients
from material_routine import material_routine 
from element_stiffness import element_stiffness
from index import assembly_index
from global_stiffness import global_stiffness
with open('mesh_1.inp', "r") as f:
        lines = f.readlines()

        num_node = int(lines[0].split()[0]) # number of nodes
        num_elem = int(lines[0].split()[1]) # number of elements
        num_fixnodes = int(lines[0].split()[2]) # number of fixed nodes
        stressState = int(lines[0].split()[3]) # if stressState == 1 plane stress; if stressState = 2 plane strain
        num_node_elem = int(lines[0].split()[4]) # number of nodes per element
        num_dof = int(lines[0].split()[5]) # degrees of freedom
        num_dim = int(lines[0].split()[6]) # dimension of a problem
        num_Gauss = int(lines[0].split()[7]) # number of integration points
        num_stress = int(lines[0].split()[8]) # number of stress comonents
        num_mat = int(lines[0].split()[9]) # number of materials 
        num_props = int(lines[0].split()[10]) # number of material properties
        
        elements = np.zeros((num_elem,num_node_elem),int)
        mat_type = np.zeros(num_elem)
        for i in range(1,num_elem+1):
            elements[i-1,0:num_node_elem] = (lines[i].split())[1:num_node_elem+1]
            mat_type[i-1] = (lines[i].split())[num_node_elem+1]
        nodes = np.zeros((num_node,num_dim),float)
        for i in range(1+num_elem,1+num_elem+num_node):
            nodes[i-1-num_elem,0] = float((lines[i].split())[1])
            nodes[i-1-num_elem,1] = float((lines[i].split())[2])
        
        fixnodes = np.zeros(num_fixnodes,int)
        fixdof = np.zeros((num_fixnodes,num_dof),int)
        disp_bc = np.zeros((num_fixnodes,num_dof),float)
        for i in range(1+num_elem+num_node,1+num_elem+num_node+num_fixnodes):
            fixnodes[i-1-num_elem-num_node] = (lines[i].split())[0]
            fixdof[i-1-num_elem-num_node,0:num_dof] = (lines[i].split())[1:num_dof+1]
            disp_bc[i-1-num_elem-num_node,0:num_dof] = (lines[i].split())[3:num_dof+3]
        
        props = np.zeros(num_props,float)
        props[0:num_props] = (lines[1+num_elem+num_node+num_fixnodes].split())[1:num_props+1]
        
        ipload = int((lines[1+num_elem+num_node+num_fixnodes+1].split())[0])
        nedge = int((lines[1+num_elem+num_node+num_fixnodes+1].split())[1])


# Call Guass quadrature points and weights using inbuilt function
PtsWts = ptwt.leggauss(num_Gauss)
points = PtsWts[0]
weights = PtsWts[1]
Points,Weights = quadrature_coefficients(num_Gauss)

# Call Stiffness tensor from material routine
mat = material_routine()
Young = props[0]
Poisson = props[1]
if stressState == 1:    
    C = mat.planestress(Young,Poisson)
elif stressState == 2:
    C = mat.planestrain(Young,Poisson)

# Initialization of global force vector and global stiffness matrix
global_K = np.zeros((num_dof*num_node, num_dof*num_node))
force = np.zeros(num_dof*num_node)

for elem in range(0,num_elem):
    elem_K  = element_stiffness(elem,num_node_elem,num_dim,num_dof,elements,nodes,num_Gauss,Points,Weights,C)
    
    index = assembly_index(elements,elem,num_dof,num_node_elem)
    
    global_K = global_stiffness(index,elem_K,global_K)
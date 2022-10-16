# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import json
import numpy as np
from input_abaqus import input_parameters
import matplotlib.pyplot as plt
from plots import plot_deformation_with_nodes,plot_field_parameter,plot_displacement_contour
with open("data_shear.json", "r") as opf:    
    disp = json.load(opf)

load_type = 1
problem = 0
Displacement = disp['0.001']
X_disp = Displacement[0:len(Displacement):2]
Y_disp = Displacement[1:len(Displacement):2]
inputs = input_parameters(load_type,problem)
num_dim, num_node_elem, nodes, num_node, elements, num_elem, num_dof, num_Gauss, num_stress = inputs.geometry_parameters()
plot_deformation_with_nodes(nodes,X_disp,Y_disp)
plot_displacement_contour(nodes,Y_disp,0.001)

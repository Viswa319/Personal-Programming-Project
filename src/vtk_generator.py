# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
from input_parameters import *
def vtk_generator(file_name,deflection,order_parameter):
    with open(file_name,"w") as output_file:
        output_file.write('# vtk DataFile Version 2.0\n')
        output_file.write('time_10.vtk\n')
        output_file.write('ASCII\n')
        output_file.write('DATASET UNSTRUCTURED_GRID\n')
        output_file.write('POINTS'+ '   ' +repr(num_node)+ '  '+ 'float \n')
        dummy = 0.0
        for i in range(0,num_node):
            output_file.write(repr(deflection[i,0]).ljust(30)+repr(deflection[i,1]).ljust(30) +repr(dummy).ljust(30)+'\n')
        if num_node_elem == 8:
            elem_type = 23
        elif num_node_elem == 4:
            elem_type = 9
        elif num_node_elem == 3:
            elem_type = 5
        output_file.write('CELLS'+ '  '+repr(num_elem)+ '  '+repr(num_elem*(num_node_elem+1))+'  \n')
        for i in range(0,num_elem):
            output_file.write(repr(num_node_elem).ljust(6)+repr(elements[i,0]).ljust(6)+repr(elements[i,1]).ljust(6) +repr(elements[i,2]).ljust(6)+repr(elements[i,3]).ljust(6)+'\n')
        output_file.write('CELL_TYPES'+ '  '+repr(num_elem)+'  \n')
        for i in range(0,num_elem):
            output_file.write(repr(elem_type)+'\n')
        output_file.write('POINT_DATA' + '  '+repr(num_node)+'\n')
        output_file.write('SCALARS Con float 1\n')
        output_file.write('LOOKUP_TABLE default\n')
        for i in range(0,num_node):
            output_file.write(repr(order_parameter[i])+'\n')
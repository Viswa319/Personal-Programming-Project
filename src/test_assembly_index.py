# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
from numpy import array_equiv
import numpy as np
from assembly_index import assembly
from input_abaqus import input_parameters
import pytest
class Test_assembly_index():
    """
    Testing indices which assemble global stiffness matrices and redidual vectors
    for both displacement and fracture phase fields.
    """
    
    def test_assmebly_index_disp_1_true(self):
        '''
        UNIT TESTING
        Aim: Test indices which assemble global stiffness matrix and global residual vector
    
        Expected result : Array of indices corresponding to global matrix for 
        displacement field of size (8)
    
        Test command : pytest test_assembly_index.py::Test_assembly_index
    
        Remarks : test case passed successfully
        '''
        load_type = 0
        problem  = 0
        inputs = input_parameters(load_type, problem)
        num_dim, num_node_elem, nodes, num_node, elements, num_elem, num_dof, num_Gauss, num_stress = inputs.geometry_parameters()
        elem = 0
        num_dof_u = 2
        assemble = assembly()
        actual_output = assemble.assembly_index_u(elem, num_dof_u, num_node_elem, elements)
        expected_output = np.array([5230,5231,5204,5205,392,393,394,395])
        assert(array_equiv(actual_output,expected_output)) is True
        
    def test_assmebly_index_disp_2_true(self):
        '''
        UNIT TESTING
        Aim: Test indices which assemble global stiffness matrix and global residual vector
    
        Expected result : Array of indices corresponding to global matrix for 
        displacement field of size (8)
    
        Test command : pytest test_assembly_index.py::Test_assembly_index
    
        Remarks : test case passed successfully
        '''
        load_type = 0
        problem  = 0
        num_dof_u = 2
        elem = 26
        inputs = input_parameters(load_type, problem)
        num_dim, num_node_elem, nodes, num_node, elements, num_elem, num_dof, num_Gauss, num_stress = inputs.geometry_parameters()
        assemble = assembly()
        
        actual_output = assemble.assembly_index_u(elem, num_dof_u, num_node_elem, elements)
        expected_output = np.array([340,341,342,343,4994,4995,5234,5235])
        assert(array_equiv(actual_output,expected_output)) is True
        
    def test_assmebly_index_phasefield_1_true(self):
        '''
        UNIT TESTING
        Aim: Test indices which assemble global stiffness matrix and global residual vector
    
        Expected result : Array of indices corresponding to global matrix for 
        phase field parameter of size (4)
    
        Test command : pytest test_assembly_index.py::Test_assembly_index
    
        Remarks : test case passed successfully
        '''
        load_type = 0
        problem  = 0
        elem = 0
        inputs = input_parameters(load_type, problem)
        num_dim, num_node_elem, nodes, num_node, elements, num_elem, num_dof, num_Gauss, num_stress = inputs.geometry_parameters()
        num_dof_phi = 1
        assemble = assembly()
        actual_output = assemble.assembly_index_phi(elem, num_dof_phi, num_node_elem, elements)
        expected_output = np.array([2615,2602,196,197])
        assert(array_equiv(actual_output,expected_output)) is True
    
    def test_assmebly_index_phasefield_2_true(self):
        '''
        UNIT TESTING
        Aim: Test indices which assemble global stiffness matrix and global residual vector
    
        Expected result : Array of indices corresponding to global matrix for 
        phase field parameter of size (4)
    
        Test command : pytest test_assembly_index.py::Test_assembly_index
    
        Remarks : test case passed successfully
        '''
        load_type = 0
        problem  = 0
        num_dof_phi = 1
        elem = 26
        inputs = input_parameters(load_type, problem)
        num_dim, num_node_elem, nodes, num_node, elements, num_elem, num_dof, num_Gauss, num_stress = inputs.geometry_parameters()
        assemble = assembly()

        actual_output = assemble.assembly_index_phi(elem, num_dof_phi, num_node_elem, elements)
        expected_output = np.array([170,171,2497,2617])
        assert(array_equiv(actual_output,expected_output)) is True
        
# test = Test_assembly_index()
# test.test_assmebly_index_disp_1_true()
# test.test_assmebly_index_phasefield_1_true()
# test.test_assmebly_index_disp_2_true()
# test.test_assmebly_index_phasefield_2_true()
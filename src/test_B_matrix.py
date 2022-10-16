# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
from numpy import array_equiv
import numpy as np
from shape_function import shape_function
from Bmatrix import Bmatrix
import pytest
class Test_B_matrix:
    """
    Testing B matrix for displacement and for fracture fields.
    """
    def test_B_matrix_disp_true(self):
        '''
        UNIT TESTING
        Aim: Test B matrix which is the strain displacement connectivity matrix in Bmatrix class
    
        Expected result : Array of B matrix of size(8,3)
    
        Test command : pytest test_B_matrix.py::Test_B_matrix
    
        Remarks : test case passed successfully
        '''
        num_node_elem = 4
        gpos = [-0.8611363,-0.8611363]
        nodes = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
        elements = np.array([1,2,3,4])
        elem_coord = nodes[elements-1,:]
        shape = shape_function(num_node_elem,gpos,elem_coord)
        dNdX = shape.get_shape_function_derivative()
        
        # Compute B matrix
        B = Bmatrix(dNdX,num_node_elem)
        actual_Bmat = B.Bmatrix_disp()
        
        expected_Bmat = np.array([[-0.46528408,0,0.46528408,0,0.03471593,0,-0.03471593,0],\
                                  [0,-0.46528408,0,-0.03471593,0,0.03471593,0,0.46528408],\
                                      [-0.46528408,-0.46528408,-0.03471593,0.46528408,0.03471593,0.03471593,0.46528408,-0.03471593]])
        
        assert(array_equiv(np.round(actual_Bmat,6),np.round(expected_Bmat,6))) is True
        
    def test_B_matrix_phi_true(self):
        '''
        UNIT TESTING
        Aim: Test B matrix for phase field parameter
    
        Expected result : Array of shape function derivatives
    
        Test command : pytest test_B_matrix.py::Test_B_matrix
    
        Remarks : test case passed successfully
        '''
        num_node_elem = 4
        gpos = [-0.8611363,-0.8611363]
        nodes = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
        elements = np.array([1,2,3,4])
        elem_coord = nodes[elements-1,:]
        shape = shape_function(num_node_elem,gpos,elem_coord)
        dNdX = shape.get_shape_function_derivative()
        
        # Compute B matrix
        B = Bmatrix(dNdX,num_node_elem)
        actual_Bmat = B.Bmatrix_phase_field()
        
        expected_Bmat = np.array([[-0.46528408,  0.46528408,  0.03471593, -0.03471593],\
                                 [-0.46528408, -0.03471593,  0.03471593,  0.46528408]])
        
        assert(array_equiv(np.round(actual_Bmat,6),np.round(expected_Bmat,6))) is True

# test = Test_B_matrix()
# test.test_B_matrix_disp_true()
# test.test_B_matrix_phi_true()
# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
from numpy import array_equiv
import numpy as np
from shape_function import shape_function
import pytest
class Test_shape_functions:
    """
    Test for shape functions, its derivatives and detreminant of Jacobian.
    """
    
    def test_shape_function_true(self):
        '''
        Test shape function
        ===================
        UNIT TESTING
        Aim: Test shape functions in shape_function class
    
        Expected result : Array of a shape function of size(4)
    
        Test command : pytest test_shape_functions.py::Test_shape_functions
    
        Remarks : test case passed successfully
        '''
        num_node_elem = 4
        gpos = [-0.8611363,-0.8611363]
        nodes = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
        elements = np.array([1,2,3,4])
        elem_coord = nodes[elements-1,:]
        shape = shape_function(num_node_elem,gpos,elem_coord)
        actual_N = shape.get_shape_function()
        expected_N = 0.25*np.array([3.46382832717769,0.2584442728223101,0.01928312717769001,0.2584442728223101])[np.newaxis]
        
        assert(array_equiv(np.round(actual_N,6),np.round(expected_N,6))) is True
    
    def test_shape_function_derivative_true(self):
        '''
        Test shape function derivative
        ==============================
        UNIT TESTING
        Aim: Test shape function derivatives in shape_function class
    
        Expected result : Array of shape function derivatives of size(2,4)
    
        Test command : pytest test_shape_functions.py::Test_shape_functions
    
        Remarks : test case passed successfully
        '''
        num_node_elem = 4
        gpos = [-0.8611363,-0.8611363]
        nodes = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
        elements = np.array([1,2,3,4])
        elem_coord = nodes[elements-1,:]
        shape = shape_function(num_node_elem,gpos,elem_coord)
        actual_dNdX = shape.get_shape_function_derivative()
        
        dNdpsi = 0.25*np.array([-1.8611363,1.8611363,0.1388637,-0.1388637])[np.newaxis]
        dNdeta = 0.25*np.array([-1.8611363,-0.1388637,0.1388637,1.8611363])[np.newaxis]
        expected_dNdX = np.r_[dNdpsi,dNdeta]
        
        assert(array_equiv(np.round(actual_dNdX,6),np.round(expected_dNdX,6))) is True
        
    def test_Jacobian_true(self):
        '''
        Test determinant of Jacobian
        ============================
        UNIT TESTING
        Aim: Test determinant of Jacobian in shape_function class
    
        Expected result : Determinant of a Jacobian matrix
    
        Test command : pytest test_shape_functions.py::Test_shape_functions
    
        Remarks : test case passed successfully
        '''
        num_node_elem = 4
        gpos = [-0.8611363,-0.8611363]
        nodes = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
        elements = np.array([1,2,3,4])
        elem_coord = nodes[elements-1,:]
        shape = shape_function(num_node_elem,gpos,elem_coord)
        actual_det_Jacobian = shape.get_det_Jacobian()
        
        J = np.array([[1,0],[0,1]])
        expected_det_Jacobian = np.linalg.det(J)
        assert(array_equiv(actual_det_Jacobian,expected_det_Jacobian)) is True
        
# test = Test_shape_functions()
# test.test_shape_function_true()
# test.test_shape_function_derivative_true()
# test.test_Jacobian_true()
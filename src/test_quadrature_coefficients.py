# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
from numpy import array_equiv
import numpy as np
from geometry import quadrature_coefficients
import pytest
class Test_quadrature_coefficients:
    """
    Testing Gauss quadrature points and weights which are used for integration.
    """
    
    def test_quadrature_points_2_true(self):
        '''
        UNIT TESTING
        Aim: Test quadrature points and weights for 2 Gauss points which is calculated using quadrature_coefficients in geometry
    
        Expected result : Array of Guass points of size (4,2)
    
        Test command : pytest test_quadrature_coefficients.py::Test_quadrature_coefficients
    
        Remarks : test case passed successfully
        '''
        actual_points,a = quadrature_coefficients(2)
        expected_points = np.array([[-0.57735027,  -0.57735027],[-0.57735027,  0.57735027],[0.57735027,  -0.57735027],[0.57735027,  0.57735027]])
        
        assert(array_equiv(np.round(actual_points,6),np.round(expected_points,6))) is True
    
    def test_quadrature_weights_2_true(self):
        '''
        UNIT TESTING
        Aim: Test quadrature points and weights for 2 Gauss points which is calculated using quadrature_coefficients in geometry
    
        Expected result : Array of weights for Guass points of size (4)
    
        Test command : pytest test_quadrature_coefficients.py::Test_quadrature_coefficients
    
        Remarks : test case passed successfully
        '''
        a,actual_weights = quadrature_coefficients(2)
        expected_weights = np.array([1,1,1,1])
        
        assert(array_equiv(actual_weights,expected_weights)) is True
        
    def test_quadrature_points_4_true(self):
        '''
        UNIT TESTING
        Aim: Test quadrature points for 4 Gauss points which is calculated using quadrature_coefficients in geometry
    
        Expected result : Array of Guass points of size(16,2)
    
        Test command : pytest test_quadrature_coefficients.py::Test_quadrature_coefficients
    
        Remarks : test case passed successfully
        '''
        
        actual_points_1,a = quadrature_coefficients(4)
        expected_points_1 = np.zeros((16,2))
        expected_points_1[0:4,0] = -0.8611363
        expected_points_1[4:8,0] = -0.3399810
        expected_points_1[8:12,0] = 0.3399810
        expected_points_1[12:16,0] = 0.8611363
        expected_points_1[0:16:4,1] = -0.8611363
        expected_points_1[1:16:4,1] = -0.3399810
        expected_points_1[2:16:4,1] = 0.3399810
        expected_points_1[3:16:4,1] = 0.8611363
        
        
        assert(array_equiv(np.round(actual_points_1,6),np.round(expected_points_1,6))) is True
    
    def test_quadrature_weights_4_true(self):
        '''
        UNIT TESTING
        Aim: Test quadrature weights for 4 Gauss points which is calculated using quadrature_coefficients in geometry
    
        Expected result : Array of weights for Gauss points of size (16)
    
        Test command : pytest test_quadrature_coefficients.py::Test_quadrature_coefficients
    
        Remarks : test case passed successfully
        '''
        
        a,actual_weights_1 = quadrature_coefficients(4)
        
        expected_weights_1 = np.zeros(16)
        expected_weights_1[0] = expected_weights_1[3] = expected_weights_1[12] = expected_weights_1[15] = 0.347855*0.347855
        expected_weights_1[1] = expected_weights_1[2] = expected_weights_1[4] = expected_weights_1[7] = 0.347855*0.652145
        expected_weights_1[8] = expected_weights_1[11] = expected_weights_1[13] = expected_weights_1[14] = 0.347855*0.652145
        expected_weights_1[5] = expected_weights_1[6] = expected_weights_1[9] = expected_weights_1[10] = 0.652145*0.652145
        
        assert(array_equiv(np.round(actual_weights_1,6),np.round(expected_weights_1,6))) is True
        
# test = Test_quadrature_coefficients()
# test.test_quadrature_points_2_true()
# test.test_quadrature_points_4_true()
# test.test_quadrature_weights_2_true()
# test.test_quadrature_weights_4_true()
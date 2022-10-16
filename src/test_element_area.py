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
class Test_element_area:
    """
    Testing area of an element, elements considered here are:\n
        square,\n
        rectamgle and\n
        parallelogram.
    
    """

    def test_element_area_square_1_true(self):
        '''
        UNIT TESTING
        Aim: Test area of a square element using determinant of Jacobian
    
        Expected result : Area of an element
    
        Test command : pytest test_element_area.py::Test_element_area
    
        Remarks : test case passed successfully
        '''
        num_node_elem = 4
        gpos = [-0.8611363,-0.8611363]
        nodes_1 = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
        elements = np.array([1,2,3,4])
        elem_coord_1 = nodes_1[elements-1,:]
        shape_1 = shape_function(num_node_elem,gpos,elem_coord_1)
        det_Jacobian_1 = shape_1.get_det_Jacobian()
        
        actual_area_1 = 4*det_Jacobian_1
        expected_area_1 = 4 
        assert(array_equiv(actual_area_1,expected_area_1)) is True
    
    def test_element_area_square_2_true(self):
        '''
        UNIT TESTING
        Aim: Test area of a square element using determinant of Jacobian
    
        Expected result : Area of an element
    
        Test command : pytest test_element_area.py::Test_element_area
    
        Remarks : test case passed successfully
        '''
        num_node_elem = 4
        gpos = [-0.8611363,0.8611363]
        elements = np.array([1,2,3,4])
        nodes_2 = np.array([[0,0],[1,0],[1,1],[0,1]])
        elem_coord_2 = nodes_2[elements-1,:]
        shape_2 = shape_function(num_node_elem,gpos,elem_coord_2)
        det_Jacobian_2 = shape_2.get_det_Jacobian()
        
        actual_area_2 = 4*det_Jacobian_2
        expected_area_2 = 1 
        assert(array_equiv(actual_area_2,expected_area_2)) is True
        
        
    def test_element_area_square_3_true(self):
        '''
        UNIT TESTING
        Aim: Test area of a square element using determinant of Jacobian
    
        Expected result : Area of an element
    
        Test command : pytest test_element_area.py::Test_element_area
    
        Remarks : test case passed successfully
        '''
        num_node_elem = 4
        gpos = [0.8611363,-0.8611363]
        elements = np.array([1,2,3,4])
        
        nodes_3 = np.array([[-2,-2],[2,-2],[2,2],[-2,2]])
        elem_coord_3 = nodes_3[elements-1,:]
        shape_3 = shape_function(num_node_elem,gpos,elem_coord_3)
        det_Jacobian_3 = shape_3.get_det_Jacobian()
        
        actual_area_3 = 4*det_Jacobian_3
        expected_area_3 = 16
        assert(array_equiv(actual_area_3,expected_area_3)) is True
    
    def test_element_area_square_4_true(self):
        '''
        UNIT TESTING
        Aim: Test area of a square element using determinant of Jacobian
    
        Expected result : Area of an element
    
        Test command : pytest test_element_area.py::Test_element_area
    
        Remarks : test case passed successfully
        '''
        num_node_elem = 4
        gpos = [-0.8611363,0.8611363]
        elements = np.array([1,2,3,4])
        
        nodes_3 = np.array([[-0.6,-0.3],[0.3,-0.3],[0.3,0.6],[-0.6,0.6]])
        elem_coord_3 = nodes_3[elements-1,:]
        shape_3 = shape_function(num_node_elem,gpos,elem_coord_3)
        det_Jacobian_3 = shape_3.get_det_Jacobian()
        actual_area_3 = 4*det_Jacobian_3
        expected_area_3 = 0.81
        assert(array_equiv(np.round(actual_area_3,4),expected_area_3)) is True
        
    def test_element_area_square_5_true(self):
        '''
        UNIT TESTING
        Aim: Test area of a square element using determinant of Jacobian
    
        Expected result : Area of an element
    
        Test command : pytest test_element_area.py::Test_element_area
    
        Remarks : test case passed successfully
        '''
        num_node_elem = 4
        gpos = [-0.8611363,0.8611363]
        elements = np.array([1,2,3,4])
        
        nodes_3 = np.array([[-0.46238986,  0.03835986],
       [-0.46565604,  0.07664392],
       [-0.5       ,  0.07692308],
       [-0.5       ,  0.03846154]])
        elem_coord_3 = nodes_3[elements-1,:]
        shape_3 = shape_function(num_node_elem,gpos,elem_coord_3)
        det_Jacobian_3 = shape_3.get_det_Jacobian()
        actual_area_3 = 4*det_Jacobian_3
        expected_area_3 = 0.0014465439
        assert(array_equiv(np.round(actual_area_3,4),np.round(expected_area_3,4))) is True
        
    def test_element_area_rectangle_1_true(self):
        '''
        UNIT TESTING
        Aim: Test area of a rectangular element using determinant of Jacobian
    
        Expected result : Area of an element
    
        Test command : pytest test_element_area.py::Test_element_area
    
        Remarks : test case passed successfully
        '''
        num_node_elem = 4
        gpos = [-0.8611363,-0.8611363]
        elements = np.array([1,2,3,4])
        
        # area of a rectangular element
        nodes_4 = np.array([[-2,-2],[1,-2],[1,2],[-2,2]])
        elem_coord_4 = nodes_4[elements-1,:]
        shape_4 = shape_function(num_node_elem,gpos,elem_coord_4)
        det_Jacobian_4 = shape_4.get_det_Jacobian()
        
        actual_area_4 = 4*np.round(det_Jacobian_4,4)
        expected_area_4 = 12
        assert(array_equiv(actual_area_4,expected_area_4)) is True
        
    def test_element_area_rectangle_2_true(self):
        '''
        UNIT TESTING
        Aim: Test area of a rectangular element using determinant of Jacobian
    
        Expected result : Area of an element
    
        Test command : pytest test_element_area.py::Test_element_area
    
        Remarks : test case passed successfully
        '''
        num_node_elem = 4
        gpos = [-0.8611363,0.8611363]
        elements = np.array([1,2,3,4])
        
        # area of a rectangular element
        nodes_4 = np.array([[0.3,0.8],[1.8,0.8],[1.8,1.6],[0.3,1.6]])
        elem_coord_4 = nodes_4[elements-1,:]
        shape_4 = shape_function(num_node_elem,gpos,elem_coord_4)
        det_Jacobian_4 = shape_4.get_det_Jacobian()
        actual_area_4 = 4*np.round(det_Jacobian_4,4)
        expected_area_4 = 1.20
        assert(array_equiv(actual_area_4,expected_area_4)) is True
        
    def test_element_area_paralellogram_1_true(self):
        '''
        UNIT TESTING
        Aim: Test area of a paralellogram element using determinant of Jacobian
    
        Expected result : Area of an element
    
        Test command : pytest test_element_area.py::Test_element_area
    
        Remarks : test case passed successfully
        '''
        num_node_elem = 4
        gpos = [0.8611363,-0.8611363]
        elements = np.array([1,2,3,4])
      
        # area of a parallelogram
        nodes_5 = np.array([[-2,-6],[4,-6],[2,-3],[-4,-3]])
        elem_coord_5 = nodes_5[elements-1,:]
        shape_5 = shape_function(num_node_elem,gpos,elem_coord_5)
        det_Jacobian_5 = shape_5.get_det_Jacobian()
        
        actual_area_5 = 4*np.round(det_Jacobian_5,4)
        expected_area_5 = 18
        assert(array_equiv(actual_area_5,expected_area_5)) is True
        
    def test_element_area_paralellogram_2_true(self):
        '''
        UNIT TESTING
        Aim: Test area of a paralellogram element using determinant of Jacobian
    
        Expected result : Area of an element
    
        Test command : pytest test_element_area.py::Test_element_area
    
        Remarks : test case passed successfully
        '''
        num_node_elem = 4
        gpos = [0.8611363,0.8611363]
        elements = np.array([1,2,3,4])
      
        # area of a parallelogram
        nodes_5 = np.array([[6,1],[19,1],[14,13],[1,13]])
        elem_coord_5 = nodes_5[elements-1,:]
        shape_5 = shape_function(num_node_elem,gpos,elem_coord_5)
        det_Jacobian_5 = shape_5.get_det_Jacobian()
        
        actual_area_5 = 4*np.round(det_Jacobian_5,4)
        expected_area_5 = 156
        assert(array_equiv(actual_area_5,expected_area_5)) is True

test = Test_element_area()
test.test_element_area_square_1_true()
test.test_element_area_square_2_true()
test.test_element_area_square_3_true()
test.test_element_area_square_4_true()
test.test_element_area_square_5_true
test.test_element_area_rectangle_1_true()
test.test_element_area_rectangle_2_true()
test.test_element_area_paralellogram_1_true()
test.test_element_area_paralellogram_2_true()

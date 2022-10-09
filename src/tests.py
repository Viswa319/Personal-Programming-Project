# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
from numpy import array,array_equiv
import numpy as np
from geometry import quadrature_coefficients
from material_routine import material_routine
from assembly_index import assembly
from shape_function import shape_function
from Bmatrix import Bmatrix
import pytest
def test_plane_stress_true():
    '''
    UNIT TESTING
    Aim: Test plane stress stiffness tensor from planestress method in material_routine class

    Expected result : Array of stiffness tensor

    Test command : pytest test.py::test_plane_stress_true

    Remarks : test case passed successfully
    '''
    Young = 1
    Poisson = 0.3
    mat = material_routine(Young,Poisson)
    actual_C = mat.planestress()
    expected_C = (Young/(1-Poisson**2))*np.array([[1,Poisson,0],[Poisson,1,0],[0,0,(1-Poisson)/2]])
    assert (array_equiv(actual_C,expected_C)) is True
    
def test_plane_strain_true():
    '''
    UNIT TESTING
    Aim: Test plane strain stiffness tensor from planestrain method in material_routine class

    Expected result : Array of stiffness tensor

    Test command : pytest test.py::test_plane_strain_true

    Remarks : test case passed successfully
    '''
    Young = 1
    Poisson = 0.3
    mat = material_routine(Young,Poisson)
    actual_C = mat.planestrain()
    expected_C = ((Young)*(1-Poisson))/((1-Poisson*2)*(1+Poisson))*np.array([[1,Poisson/(1-Poisson),0],[Poisson/(1-Poisson),1,0],[0,0,(1-2*Poisson)/(2*(1-Poisson))]])
    assert (array_equiv(actual_C,expected_C)) is True
    
def test_quadrature_coefficients_true():
    '''
    UNIT TESTING
    Aim: Test quadrature points and weights which is calculated using quadrature_coefficients in geometry

    Expected result : Array of Guass points and weights

    Test command : pytest test.py::test_quadrature_coefficients_true()

    Remarks : test case passed successfully
    '''
    actual_points,actual_weights = quadrature_coefficients(2)
    expected_points = np.array([[-0.57735027,  -0.57735027],[-0.57735027,  0.57735027],[0.57735027,  -0.57735027],[0.57735027,  0.57735027]])
    expected_weights = np.array([1,1,1,1])
    
    assert(array_equiv(np.round(actual_points,6),np.round(expected_points,6))) is True
    assert(array_equiv(actual_weights,expected_weights)) is True
    
    actual_points_1,actual_weights_1 = quadrature_coefficients(4)
    expected_points_1 = np.zeros((16,2))
    expected_points_1[0:4,0] = -0.8611363
    expected_points_1[4:8,0] = -0.3399810
    expected_points_1[8:12,0] = 0.3399810
    expected_points_1[12:16,0] = 0.8611363
    expected_points_1[0:16:4,1] = -0.8611363
    expected_points_1[1:16:4,1] = -0.3399810
    expected_points_1[2:16:4,1] = 0.3399810
    expected_points_1[3:16:4,1] = 0.8611363
    
    expected_weights_1 = np.zeros(16)
    expected_weights_1[0] = expected_weights_1[3] = expected_weights_1[12] = expected_weights_1[15] = 0.347855*0.347855
    expected_weights_1[1] = expected_weights_1[2] = expected_weights_1[4] = expected_weights_1[7] = 0.347855*0.652145
    expected_weights_1[8] = expected_weights_1[11] = expected_weights_1[13] = expected_weights_1[14] = 0.347855*0.652145
    expected_weights_1[5] = expected_weights_1[6] = expected_weights_1[9] = expected_weights_1[10] = 0.652145*0.652145
    
    assert(array_equiv(np.round(actual_weights_1,6),np.round(expected_weights_1,6))) is True
    assert(array_equiv(np.round(actual_points_1,6),np.round(expected_points_1,6))) is True
    
def test_shape_function_true():
    '''
    UNIT TESTING
    Aim: Test shape functions in shape_function class

    Expected result : Array of a shape function

    Test command : pytest test.py::test_shape_function_true()

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
def test_B_matrix_true():
    '''
    UNIT TESTING
    Aim: Test B matrix which is the strain displacement connectivity matrix in Bmatrix class

    Expected result : Array of B matrix

    Test command : pytest test.py::test_B_matrix_true()

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
    
def test_shape_function_derivative_true():
    '''
    UNIT TESTING
    Aim: Test shape function derivatives in shape_function class

    Expected result : Array of shape function derivatives

    Test command : pytest test.py::test_shape_function_derivative_true()

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
    
def test_Jacobian_true():
    '''
    UNIT TESTING
    Aim: Test determinant of Jacobian in shape_function class

    Expected result : Determinant of a Jacobian matrix

    Test command : pytest test.py::test_Jacobian_true()

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
    
def test_element_area_true():
    '''
    UNIT TESTING
    Aim: Test area of an element using determinant of Jacobian

    Expected result : Area of an element

    Test command : pytest test.py::test_element_area_true()

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
    
    nodes_2 = np.array([[0,0],[1,0],[1,1],[0,1]])
    elem_coord_2 = nodes_2[elements-1,:]
    shape_2 = shape_function(num_node_elem,gpos,elem_coord_2)
    det_Jacobian_2 = shape_2.get_det_Jacobian()
    
    actual_area_2 = 4*det_Jacobian_2
    expected_area_2 = 1 
    assert(array_equiv(actual_area_2,expected_area_2)) is True
    
    nodes_3 = np.array([[-2,-2],[2,-2],[2,2],[-2,2]])
    elem_coord_3 = nodes_3[elements-1,:]
    shape_3 = shape_function(num_node_elem,gpos,elem_coord_3)
    det_Jacobian_3 = shape_3.get_det_Jacobian()
    
    actual_area_3 = 4*det_Jacobian_3
    expected_area_3 = 16
    assert(array_equiv(actual_area_3,expected_area_3)) is True

def test_assembly_index_true():
    '''
    UNIT TESTING
    Aim: Test area of an element using determinant of Jacobian

    Expected result : Area of an element

    Test command : pytest test.py::test_element_area_true()

    Remarks : test case passed successfully
    '''
    for elem in range(0,num_elem):
        elem_stiff = element_staggered(elem,Points,Weights,disp,phi,stress,strain,strain_energy,elements,nodes,num_dof,num_node_elem,num_Gauss_2D)
        K_uu = elem_stiff.element_stiffness_displacement(C,k_const)
        assemble = assembly()
        
        index_u = assemble.assembly_index_u(elem,num_dof_u,num_node_elem,elements)
        
        X,Y = np.meshgrid(index_u,index_u,sparse=True)
        global_K_disp[X,Y] =  global_K_disp[X,Y] + K_uu
        del X,Y
test_plane_stress_true()
test_plane_strain_true()
test_quadrature_coefficients_true()
test_shape_function_true()
test_B_matrix_true()
test_shape_function_derivative_true()
test_Jacobian_true()
test_element_area_true()
test_assembly_index_true()
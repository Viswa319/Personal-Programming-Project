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
from input_abaqus import input_parameters
import pytest
from element_staggered import element_staggered
from shape_function import shape_function
from Bmatrix import Bmatrix
from assembly_index import assembly

elem = 0
load_type = 0
problem = 0
inputs = input_parameters(load_type,problem)
 
# num_dim, num_node_elem, nodes, num_node, elements, num_elem, num_dof, num_Gauss, num_stress = inputs.geometry_parameters()
k_const,G_c,l_0,Young,Poisson,stressState = inputs.material_parameters_elastic()
num_step, max_iter, max_tol, disp_inc = inputs.time_integration_parameters()
num_Gauss = 2
num_dim = 2
num_dof = 3
num_stress = 3
num_node_elem = 4
num_Gauss_2D = num_Gauss**num_dim
nodes = np.array([[0,0],[1,0],[1,1],[0,1]])
num_node = len(nodes)
elements = np.array([[1,2,3,4]])
num_elem = len(elements)

# the number of DOFs for displacements per node
num_dof_u = num_dof-1

# the number of DOFs for order parameters per node
num_dof_phi = num_dof-2
    
# total number of variables for displacements
num_tot_var_u = num_node*num_dof_u
    
# total number of variables for displacements
num_tot_var_phi = num_node*num_dof_phi
# Initialize stress component values at the integration points of all elements
stress = np.zeros((num_elem,num_Gauss_2D,num_stress))
    
# Initialize strain component values at the integration points of all elements 
strain = np.zeros((num_elem,num_Gauss_2D,num_stress))
    
# Initialize strain energy 
strain_energy = np.zeros((num_elem,num_Gauss_2D))
    
# Initialize displacment values and displacement boundary conditions
disp = np.zeros(num_tot_var_u)
disp_bc = np.zeros(num_tot_var_u)
    
# Initialize phi (field parameter) values
phi = np.zeros(num_tot_var_phi)

# Call Guass quadrature points and weights using inbuilt function
Points,Weights = quadrature_coefficients(num_Gauss)

# Call Stiffness tensor from material routine
mat = material_routine(problem)
C = mat.material_elasticity()

elem_stiff = element_staggered(elem,Points,Weights,disp,phi,stress,strain_energy,elements,nodes,num_dof,num_node_elem,num_Gauss_2D)
actual_K_phiphi, residual_elem_phi = elem_stiff.element_stiffness_field_parameter(G_c,l_0)
actual_K_uu, F_int_elem = elem_stiff.element_stiffness_displacement(C,k_const)

# Boundary points for all edges
bot = np.where(nodes[:,1] == min(nodes[:,1]))[0] # bottom edge nodes
left = np.where(nodes[:,0] == min(nodes[:,0]))[0] # left edge nodes
right = np.where(nodes[:,0] == max(nodes[:,0]))[0] # right edge nodes
top = np.where(nodes[:,1] == max(nodes[:,1]))[0] # top edge nodes

# Getting the fixed degrees of freedom at all nodes
# If boundary condition is prescribed at certain node, then it is given value 1 or else 0
fixed_dof = np.zeros(num_tot_var_u)
fixed_dof[(top*2)+1] = 1
fixed_dof[(top*2)] = 1
fixed_dof[(bot*2)+1] = 1
fixed_dof[bot*2] = 1

# Initialization of global force vector and global stiffness matrix for  displacement
global_force_disp = np.zeros(num_tot_var_u)
global_K_disp = np.zeros((num_tot_var_u, num_tot_var_u))
for elem in range(0,num_elem):
    elem_stiff = element_staggered(elem,Points,Weights,disp,phi,stress,strain_energy,elements,nodes,num_dof,num_node_elem,num_Gauss_2D)
    actual_K_uu, F_int_elem = elem_stiff.element_stiffness_displacement(C,k_const)
    assemble = assembly()
    
    index_u = assemble.assembly_index_u(elem,num_dof_u,num_node_elem,elements)
    
    X,Y = np.meshgrid(index_u,index_u,sparse=True)
    global_K_disp[X,Y] =  global_K_disp[X,Y] + actual_K_uu
    del X,Y
elem_coord = nodes
def test_shape_function_1_true():
    '''
    UNIT TESTING
    Aim: Test shape functions in shape_function class

    Expected result : Array of shape function

    Test command : pytest test_one_element.py::test_shape_function_1_true()

    Remarks : test case passed successfully
    '''
    gpos_1 = Points[0]
    shape_1 = shape_function(num_node_elem,gpos_1,elem_coord)
    actual_N_1 = shape_1.get_shape_function()
    expected_N_1 = array([[0.62200847, 0.16666667, 0.0446582 , 0.16666667]])
    assert(array_equiv(np.round(actual_N_1,6),np.round(expected_N_1,6))) is True

def test_shape_function_2_true():
    '''
    UNIT TESTING
    Aim: Test shape functions in shape_function class

    Expected result : Array of shape function

    Test command : pytest test_one_element.py::test_shape_function_2_true()

    Remarks : test case passed successfully
    '''
    gpos_2 = Points[1]
    shape_2 = shape_function(num_node_elem,gpos_2,elem_coord)
    actual_N_2 = shape_2.get_shape_function()
    expected_N_2 = array([[0.16666667, 0.0446582 , 0.16666667, 0.62200847]])
    assert(array_equiv(np.round(actual_N_2,6),np.round(expected_N_2,6))) is True
    
def test_shape_function_3_true():
    '''
    UNIT TESTING
    Aim: Test shape functions in shape_function class

    Expected result : Array of shape function

    Test command : pytest test_one_element.py::test_shape_function_3_true()

    Remarks : test case passed successfully
    '''
    gpos_3 = Points[2]
    shape_3 = shape_function(num_node_elem,gpos_3,elem_coord)
    actual_N_3 = shape_3.get_shape_function()
    expected_N_3 = array([[0.16666667, 0.62200847, 0.16666667, 0.0446582 ]])
    assert(array_equiv(np.round(actual_N_3,6),np.round(expected_N_3,6))) is True
    
def test_shape_function_4_true():
    '''
    UNIT TESTING
    Aim: Test shape functions in shape_function class

    Expected result : Array of shape function

    Test command : pytest test_one_element.py::test_shape_function_4_true()

    Remarks : test case passed successfully
    '''
    gpos_4 = Points[3]
    shape_4 = shape_function(num_node_elem,gpos_4,elem_coord)
    actual_N_4 = shape_4.get_shape_function()
    expected_N_4 = array([[0.0446582 , 0.16666667, 0.62200847, 0.16666667]])
    assert(array_equiv(np.round(actual_N_4,6),np.round(expected_N_4,6))) is True

def test_shape_function_derivative_1_true():
    '''
    UNIT TESTING
    Aim: Test shape function derivatives in shape_function class

    Expected result : Array of shape function derivative

    Test command : pytest test_one_element.py::test_shape_function_derivative_1_true()

    Remarks : test case passed successfully
    '''
    gpos_1 = Points[0]
    shape_1 = shape_function(num_node_elem,gpos_1,elem_coord)
    actual_dNdX_1 = shape_1.get_shape_function_derivative()
    expected_dNdX_1 = array([[-0.78867513,  0.78867513,  0.21132487, -0.21132487],
       [-0.78867513, -0.21132487,  0.21132487,  0.78867513]])
    assert(array_equiv(np.round(actual_dNdX_1,6),np.round(expected_dNdX_1,6))) is True
    
def test_shape_function_derivative_2_true():
    '''
    UNIT TESTING
    Aim: Test shape function derivatives in shape_function class

    Expected result : Array of shape function derivative

    Test command : pytest test_one_element.py::test_shape_function_derivative_2_true()

    Remarks : test case passed successfully
    '''
    gpos_2 = Points[1]
    shape_2 = shape_function(num_node_elem,gpos_2,elem_coord)
    actual_dNdX_2 = shape_2.get_shape_function_derivative()
    expected_dNdX_2 = array([[-0.21132487,  0.21132487,  0.78867513, -0.78867513],
       [-0.78867513, -0.21132487,  0.21132487,  0.78867513]])
    assert(array_equiv(np.round(actual_dNdX_2,6),np.round(expected_dNdX_2,6))) is True

def test_shape_function_derivative_3_true():
    '''
    UNIT TESTING
    Aim: Test shape function derivatives in shape_function class

    Expected result : Array of shape function derivative

    Test command : pytest test_one_element.py::test_shape_function_derivative_3_true()

    Remarks : test case passed successfully
    '''
    gpos_3 = Points[2]
    shape_3 = shape_function(num_node_elem,gpos_3,elem_coord)
    actual_dNdX_3 = shape_3.get_shape_function_derivative()
    expected_dNdX_3 = array([[-0.78867513,  0.78867513,  0.21132487, -0.21132487],
       [-0.21132487, -0.78867513,  0.78867513,  0.21132487]])
    assert(array_equiv(np.round(actual_dNdX_3,6),np.round(expected_dNdX_3,6))) is True
    
def test_shape_function_derivative_4_true():
    '''
    UNIT TESTING
    Aim: Test shape function derivatives in shape_function class

    Expected result : Array of shape function derivative

    Test command : pytest test_one_element.py::test_shape_function_derivative_4_true()

    Remarks : test case passed successfully
    '''
    gpos_4 = Points[3]
    shape_4 = shape_function(num_node_elem,gpos_4,elem_coord)
    actual_dNdX_4 = shape_4.get_shape_function_derivative()
    expected_dNdX_4 = array([[-0.21132487,  0.21132487,  0.78867513, -0.78867513],
       [-0.21132487, -0.78867513,  0.78867513,  0.21132487]])
    assert(array_equiv(np.round(actual_dNdX_4,6),np.round(expected_dNdX_4,6))) is True
    
def test_det_Jacobian_1_true():
    '''
    UNIT TESTING
    Aim: Test determinant of Jacobian in shape_function class

    Expected result : Determinant of a Jacobian

    Test command : pytest test_one_element.py::test_det_Jacobian_1_true()

    Remarks : test case passed successfully
    '''
    gpos = Points[0]
    shape = shape_function(num_node_elem,gpos,elem_coord)
    actual_det_Jacobian = shape.get_det_Jacobian()
    expected_det_Jacobian = 0.25
    assert(array_equiv(actual_det_Jacobian,expected_det_Jacobian)) is True
    
def test_det_Jacobian_2_true():
    '''
    UNIT TESTING
    Aim: Test determinant of Jacobian in shape_function class

    Expected result : Determinant of a Jacobian

    Test command : pytest test_one_element.py::test_det_Jacobian_2_true()

    Remarks : test case passed successfully
    '''
    gpos = Points[1]
    shape = shape_function(num_node_elem,gpos,elem_coord)
    actual_det_Jacobian = shape.get_det_Jacobian()
    expected_det_Jacobian = 0.25
    assert(array_equiv(actual_det_Jacobian,expected_det_Jacobian)) is True
    
def test_det_Jacobian_3_true():
    '''
    UNIT TESTING
    Aim: Test determinant of Jacobian in shape_function class

    Expected result : Determinant of a Jacobian

    Test command : pytest test_one_element.py::test_det_Jacobian_3_true()

    Remarks : test case passed successfully
    '''    
    gpos = Points[2]
    shape = shape_function(num_node_elem,gpos,elem_coord)
    actual_det_Jacobian = shape.get_det_Jacobian()
    expected_det_Jacobian = 0.25
    assert(array_equiv(actual_det_Jacobian,expected_det_Jacobian)) is True
    
def test_det_Jacobian_4_true():
    '''
    UNIT TESTING
    Aim: Test determinant of Jacobian in shape_function class

    Expected result : Determinant of a Jacobian

    Test command : pytest test_one_element.py::test_det_Jacobian_4_true()

    Remarks : test case passed successfully
    '''
    gpos = Points[3]
    shape = shape_function(num_node_elem,gpos,elem_coord)
    actual_det_Jacobian = shape.get_det_Jacobian()
    expected_det_Jacobian = 0.25
    assert(array_equiv(actual_det_Jacobian,expected_det_Jacobian)) is True

def test_det_Jacobian_equal_true():
    '''
    UNIT TESTING
    Aim: Test determinant of Jacobian, determinant should be same for all Gauss points in shape_function class

    Expected result : Determinant of a Jacobian equal for all Gauss points

    Test command : pytest test_one_element.py::test_det_Jacobian_equal_true()

    Remarks : test case passed successfully
    '''
    gpos_1 = Points[0]
    shape_1 = shape_function(num_node_elem,gpos_1,elem_coord)
    actual_det_Jacobian_1 = shape_1.get_det_Jacobian()

    gpos_2 = Points[1]
    shape_2 = shape_function(num_node_elem,gpos_2,elem_coord)
    actual_det_Jacobian_2 = shape_2.get_det_Jacobian()

    gpos_3 = Points[2]
    shape_3 = shape_function(num_node_elem,gpos_3,elem_coord)
    actual_det_Jacobian_3 = shape_3.get_det_Jacobian()

    gpos_4 = Points[3]
    shape_4 = shape_function(num_node_elem,gpos_4,elem_coord)
    actual_det_Jacobian_4 = shape_4.get_det_Jacobian()
    expected_det_Jacobian = 0.25
    
    assert(array_equiv(actual_det_Jacobian_1, actual_det_Jacobian_2)) is True
    assert(array_equiv(actual_det_Jacobian_2, actual_det_Jacobian_3)) is True     
    assert(array_equiv(actual_det_Jacobian_3, actual_det_Jacobian_4)) is True
    assert(array_equiv(actual_det_Jacobian_4, actual_det_Jacobian_1)) is True  
    assert(array_equiv(actual_det_Jacobian_1, expected_det_Jacobian)) is True  
    
def test_element_area_true():
    '''
    UNIT TESTING
    Aim: Test area of an element using determinant of Jacobian

    Expected result : Area of an element

    Test command : pytest test_one_element.py::test_element_area_true()

    Remarks : test case passed successfully
    '''
    gpos = Points[0]
    elem_coord = nodes
    shape = shape_function(num_node_elem,gpos,elem_coord)
    det_Jacobian = shape.get_det_Jacobian()
    
    actual_area = 4*det_Jacobian
    expected_area = 1

    assert(array_equiv(actual_area,expected_area)) is True
    
def test_B_matrix_phasefield_1_true():
    '''
    UNIT TESTING
    Aim: Test B matrix for fiels order parameter which is derivative of shape function

    Expected result : Array of B matrix

    Test command : pytest test_one_element.py::test_B_matrix_phasefield_1_true()

    Remarks : test case passed successfully
    '''
    gpos = Points[0]
    shape = shape_function(num_node_elem,gpos,elem_coord)
    dNdX = shape.get_shape_function_derivative()
    B = Bmatrix(dNdX,num_node_elem)
    actual_Bmat = B.Bmatrix_phase_field()
    expected_Bmat = array([[-0.78867513,  0.78867513,  0.21132487, -0.21132487],
       [-0.78867513, -0.21132487,  0.21132487,  0.78867513]])
    assert(array_equiv(np.round(actual_Bmat,6),np.round(expected_Bmat,6))) is True
    
def test_B_matrix_phasefield_2_true():
    '''
    UNIT TESTING
    Aim: Test B matrix for fiels order parameter which is derivative of shape function

    Expected result : Array of B matrix

    Test command : pytest test_one_element.py::test_B_matrix_phasefield_2_true()

    Remarks : test case passed successfully
    '''
    gpos = Points[1]
    shape = shape_function(num_node_elem,gpos,elem_coord)
    dNdX = shape.get_shape_function_derivative()
    B = Bmatrix(dNdX,num_node_elem)
    actual_Bmat = B.Bmatrix_phase_field()
    expected_Bmat = array([[-0.21132487,  0.21132487,  0.78867513, -0.78867513],
       [-0.78867513, -0.21132487,  0.21132487,  0.78867513]])
    assert(array_equiv(np.round(actual_Bmat,6),np.round(expected_Bmat,6))) is True
    
def test_B_matrix_phasefield_3_true():
    '''
    UNIT TESTING
    Aim: Test B matrix for fiels order parameter which is derivative of shape function

    Expected result : Array of B matrix

    Test command : pytest test_one_element.py::test_B_matrix_phasefield_3_true()

    Remarks : test case passed successfully
    '''
    gpos = Points[2]
    shape = shape_function(num_node_elem,gpos,elem_coord)
    dNdX = shape.get_shape_function_derivative()
    B = Bmatrix(dNdX,num_node_elem)
    actual_Bmat = B.Bmatrix_phase_field()
    expected_Bmat = array([[-0.78867513,  0.78867513,  0.21132487, -0.21132487],
       [-0.21132487, -0.78867513,  0.78867513,  0.21132487]])
    assert(array_equiv(np.round(actual_Bmat,6),np.round(expected_Bmat,6))) is True
    
def test_B_matrix_phasefield_4_true():
    '''
    UNIT TESTING
    Aim: Test B matrix for fiels order parameter which is derivative of shape function

    Expected result : Array of B matrix

    Test command : pytest test_one_element.py::test_B_matrix_phasefield_4_true()

    Remarks : test case passed successfully
    '''
    gpos = Points[3]
    shape = shape_function(num_node_elem,gpos,elem_coord)
    dNdX = shape.get_shape_function_derivative()
    B = Bmatrix(dNdX,num_node_elem)
    actual_Bmat = B.Bmatrix_phase_field()
    expected_Bmat = array([[-0.21132487,  0.21132487,  0.78867513, -0.78867513],
       [-0.21132487, -0.78867513,  0.78867513,  0.21132487]])
    assert(array_equiv(np.round(actual_Bmat,6),np.round(expected_Bmat,6))) is True
    
def test_stiffness_phasefield_true():
    '''
    UNIT TESTING
    Aim: Test stiffness matrix of order parameter.

    Expected result : Array of stiffness matrix for phase field order parameter

    Test command : pytest test_one_element.py::test_stiffness_phasefield_true()

    Remarks : test case passed successfully
    '''
    
    expected_K_phiphi = array([[7.572, 3.732, 1.839, 3.732],
       [3.732, 7.572, 3.732, 1.839],
       [1.839, 3.732, 7.572, 3.732],
       [3.732, 1.839, 3.732, 7.572]])
    assert(array_equiv(np.round(actual_K_phiphi,6),np.round(expected_K_phiphi,6))) is True

def test_eigen_values_stiffness_phasefield_positive_true():
    '''
    UNIT TESTING
    Aim: Test sign of eigen values of stiffness matrix for field order parameter

    Expected result : Array of sign of eigen values

    Test command : pytest test_one_element.py::test_eigen_values_stiffness_phasefield_positive_true()

    Remarks : test case passed successfully
    '''
    eigen_values = np.linalg.eigvalsh(actual_K_phiphi)
    actual_output = np.sign(eigen_values)
    expected_output = np.ones(num_tot_var_phi)
    assert(array_equiv(actual_output,expected_output)) is True
    
def test_stiffness_phasefield_symmetry():
    '''
    UNIT TESTING
    Aim: Test symmetry of a stiffness matrix 

    Expected result : Array which has size of stiffness matrix consists of all True 

    Test command : pytest test.py::test_stiffness_phasefield_symmetry()

    Remarks : test case passed successfully
    '''
    actual_output = np.isclose(actual_K_phiphi,np.transpose(actual_K_phiphi))
    expected_output = True*np.ones(np.shape(actual_K_phiphi))
    assert(array_equiv(actual_output,expected_output)) is True

def test_B_matrix_disp_1_true():
    '''
    UNIT TESTING
    Aim: Test B matrix for displacement which is the strain displacement connectivity matrix

    Expected result : Array of B matrix for displacement

    Test command : pytest test_one_element.py::test_B_matrix_1_true()

    Remarks : test case passed successfully
    '''
    gpos = Points[0]
    shape = shape_function(num_node_elem,gpos,elem_coord)
    dNdX = shape.get_shape_function_derivative()
    B = Bmatrix(dNdX,num_node_elem)
    actual_Bmat = B.Bmatrix_disp()
    expected_Bmat = array([[-0.78867513, 0, 0.78867513, 0, 0.21132487, 0, -0.21132487, 0],
       [0, -0.78867513, 0, -0.21132487, 0, 0.21132487, 0, 0.78867513], 
       [-0.78867513, -0.78867513, -0.21132487, 0.78867513, 0.21132487, 0.21132487, 0.78867513, -0.21132487]])
    assert(array_equiv(np.round(actual_Bmat,6),np.round(expected_Bmat,6))) is True
    
def test_B_matrix_disp_2_true():
    '''
    UNIT TESTING
    Aim: Test B matrix for displacement which is the strain displacement connectivity matrix

    Expected result : Array of B matrix for displacement

    Test command : pytest test_one_element.py::test_B_matrix_2_true()

    Remarks : test case passed successfully
    '''
    gpos = Points[1]
    shape = shape_function(num_node_elem,gpos,elem_coord)
    dNdX = shape.get_shape_function_derivative()
    B = Bmatrix(dNdX,num_node_elem)
    actual_Bmat = B.Bmatrix_disp()
    expected_Bmat = array([[-0.21132487, 0, 0.21132487, 0, 0.78867513, 0, -0.78867513, 0],
       [0, -0.78867513, 0, -0.21132487, 0, 0.21132487, 0, 0.78867513],
       [-0.78867513, -0.21132487, -0.21132487, 0.21132487, 0.21132487, 0.78867513, 0.78867513, -0.78867513]])
    assert(array_equiv(np.round(actual_Bmat,6),np.round(expected_Bmat,6))) is True
    
def test_B_matrix_disp_3_true():
    '''
    UNIT TESTING
    Aim: Test B matrix for displacement which is the strain displacement connectivity matrix

    Expected result : Array of B matrix for displacement

    Test command : pytest test_one_element.py::test_B_matrix_3_true()

    Remarks : test case passed successfully
    '''
    gpos = Points[2]
    shape = shape_function(num_node_elem,gpos,elem_coord)
    dNdX = shape.get_shape_function_derivative()
    B = Bmatrix(dNdX,num_node_elem)
    actual_Bmat = B.Bmatrix_disp()
    expected_Bmat = array([[-0.78867513, 0, 0.78867513, 0, 0.21132487, 0, -0.21132487, 0],
       [0, -0.21132487, 0, -0.78867513, 0, 0.78867513, 0, 0.21132487],
       [-0.21132487, -0.78867513, -0.78867513, 0.78867513, 0.78867513, 0.21132487, 0.21132487, -0.21132487]])
    assert(array_equiv(np.round(actual_Bmat,6),np.round(expected_Bmat,6))) is True
    
def test_B_matrix_disp_4_true():
    
    '''
    UNIT TESTING
    Aim: Test B matrix for displacement which is the strain displacement connectivity matrix

    Expected result : Array of B matrix for displacement

    Test command : pytest test_one_element.py::test_B_matrix_4_true()

    Remarks : test case passed successfully
    '''
    gpos = Points[3]
    shape = shape_function(num_node_elem,gpos,elem_coord)
    dNdX = shape.get_shape_function_derivative()
    B = Bmatrix(dNdX,num_node_elem)
    actual_Bmat = B.Bmatrix_disp()
    expected_Bmat = array([[-0.21132487, 0, 0.21132487, 0, 0.78867513, 0, -0.78867513, 0],
       [0, -0.21132487, 0, -0.78867513, 0, 0.78867513, 0, 0.21132487],
       [-0.21132487, -0.21132487, -0.78867513, 0.21132487, 0.78867513, 0.78867513, 0.21132487, -0.78867513]])
    assert(array_equiv(np.round(actual_Bmat,6),np.round(expected_Bmat,6))) is True

def test_stiffness_disp_true():
    '''
    UNIT TESTING
    Aim: Test stiffness matrix for displacement

    Expected result : Array of stiffness matrix 

    Test command : pytest test_one_element.py::test_stiffness_disp_true()

    Remarks : test case passed successfully
    '''
    expected_K_uu = array([[121153.85826923,  50480.77427885, -80769.23884615,
         10096.15485577, -60576.92913462, -50480.77427885,
         20192.30971154, -10096.15485577],
       [ 50480.77427885, 121153.85826923, -10096.15485577,
         20192.30971154, -50480.77427885, -60576.92913462,
         10096.15485577, -80769.23884615],
       [-80769.23884615, -10096.15485577, 121153.85826923,
        -50480.77427885,  20192.30971154,  10096.15485577,
        -60576.92913462,  50480.77427885],
       [ 10096.15485577,  20192.30971154, -50480.77427885,
        121153.85826923, -10096.15485577, -80769.23884615,
         50480.77427885, -60576.92913462],
       [-60576.92913462, -50480.77427885,  20192.30971154,
        -10096.15485577, 121153.85826923,  50480.77427885,
        -80769.23884615,  10096.15485577],
       [-50480.77427885, -60576.92913462,  10096.15485577,
        -80769.23884615,  50480.77427885, 121153.85826923,
        -10096.15485577,  20192.30971154],
       [ 20192.30971154,  10096.15485577, -60576.92913462,
         50480.77427885, -80769.23884615, -10096.15485577,
        121153.85826923, -50480.77427885],
       [-10096.15485577, -80769.23884615,  50480.77427885,
        -60576.92913462,  10096.15485577,  20192.30971154,
        -50480.77427885, 121153.85826923]])
    
    assert(array_equiv(np.round(actual_K_uu,6),np.round(expected_K_uu,6))) is True

def test_stiffness_disp_symmetry():
    '''
    UNIT TESTING
    Aim: Test symmetry of a stiffness matrix

    Expected result : Array which has size of stiffness matrix consists of all True 

    Test command : pytest test_one_element.py::test_element_area_true()

    Remarks : test case passed successfully
    '''
    actual_output = np.isclose(actual_K_uu,np.transpose(actual_K_uu))
    expected_output = True*np.ones(np.shape(actual_K_uu))
    assert(array_equiv(actual_output,expected_output)) is True

def test_eigen_values_stiffness_disp_positive_true():
    '''
    UNIT TESTING
    Aim: Test sign of eigen values of stiffness matrix for field order parameter

    Expected result : Array of sign of eigen values

    Test command : pytest test_one_element.py::test_element_area_true()

    Remarks : test case passed successfully
    '''
    eigen_values = np.linalg.eigvalsh(actual_K_phiphi)
    actual_output = np.sign(np.round(eigen_values,8))
    expected_output = np.ones(num_tot_var_phi)
    assert(array_equiv(actual_output,expected_output)) is True

def test_global_assembly_stiffness_disp_true():
    '''
    UNIT TESTING
    Aim: Test assembly of a global stiffness matrix

    Expected result : Array of global stiffness matrix

    Test command : pytest test_one_element.py::test_global_assembly_stiffness_disp_true()

    Remarks : test case passed successfully
    '''
    assert(array_equiv(np.round(actual_K_uu,6),np.round(global_K_disp,6))) is True
    
def test_assmebly_index_disp_true():
    '''
    UNIT TESTING
    Aim: Test indices which assemble global stiffness matrix and global residual vector

    Expected result : Array of indices corresponding to global matrix for displacement field

    Test command : pytest test_one_element.py::test_assmebly_index_disp_true()

    Remarks : test case passed successfully
    '''
    assemble = assembly()
    actual_output = assemble.assembly_index_u(elem, num_dof_u, num_node_elem, elements)
    expected_output = np.array([0,1,2,3,4,5,6,7])
    assert(array_equiv(actual_output,expected_output)) is True
    
def test_assmebly_index_phasefield_true():
    '''
    UNIT TESTING
    Aim: Test indices which assemble global stiffness matrix and global residual vector

    Expected result : Array of indices corresponding to global matrix for phase field parameter

    Test command : pytest test_one_element.py::test_assmebly_index_disp_true()

    Remarks : test case passed successfully
    '''
    assemble = assembly()
    actual_output = assemble.assembly_index_phi(elem, num_dof_phi, num_node_elem, elements)
    expected_output = np.array([0,1,2,3])
    assert(array_equiv(actual_output,expected_output)) is True

def test_material_routine_elastic_true():
    '''
    UNIT TESTING
    Aim: Test material routine for elastic case in material_routine class

    Expected result : Array of stiffness tensor for plane strain case

    Test command : pytest test_one_element.py::test_material_routine_true

    Remarks : test case passed successfully
    '''
    Young = 210000
    Poisson = 0.3
    actual_C = C
    expected_C = ((Young)*(1-Poisson))/((1-Poisson*2)*(1+Poisson))*np.array([[1,Poisson/(1-Poisson),0],[Poisson/(1-Poisson),1,0],[0,0,(1-2*Poisson)/(2*(1-Poisson))]])
    assert (array_equiv(actual_C,expected_C)) is True

# test_stiffness_phasefield_true()
# test_stiffness_phasefield_symmetry()
# test_stiffness_disp_true()
# test_eigen_values_stiffness_phasefield_positive_true()
# test_shape_function_1_true()
# test_shape_function_2_true()
# test_shape_function_3_true()
# test_shape_function_4_true()
# test_shape_function_derivative_1_true()
# test_shape_function_derivative_2_true()
# test_shape_function_derivative_3_true()
# test_shape_function_derivative_4_true()
# test_det_Jacobian_1_true()
# test_det_Jacobian_2_true()
# test_det_Jacobian_3_true()
# test_det_Jacobian_4_true()
# test_det_Jacobian_equal_true
# test_element_area_true()
# test_B_matrix_phasefield_1_true()
# test_B_matrix_phasefield_2_true()
# test_B_matrix_phasefield_3_true()
# test_B_matrix_phasefield_4_true()
# test_stiffness_disp_symmetry()
# test_B_matrix_disp_1_true()
# test_B_matrix_disp_2_true()
# test_B_matrix_disp_3_true()
# test_B_matrix_disp_4_true()
# test_eigen_values_stiffness_disp_positive_true()
# test_global_assembly_stiffness_disp_true()
# test_assmebly_index_disp_true()
# test_assmebly_index_phasefield_true()
# test_material_routine_elastic_true()
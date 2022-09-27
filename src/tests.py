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
from Lagrange_interpolant import Lagrange_interpolant
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
    Aim: Test plane stress stiffness tensor from planestrain method in material_routine class

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

test_plane_stress_true()
test_plane_strain_true()
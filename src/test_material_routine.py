# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
from numpy import array_equiv
import numpy as np
from material_routine import material_routine
class Test_material_routine:
    """
    Testing material routine class by testing plane stress and plane strain cases,
    and testing elastic and elastic-plastic material routine case.
    """
    
    def test_plane_stress_true(self):
        '''
        UNIT TESTING
        Aim: Test plane stress stiffness tensor from planestress method in material_routine class
    
        Expected result : Array of stiffness tensor of size (3,3)
    
        Test command : pytest test_material_routine.py::Test_material_routine
    
        Remarks : test case passed successfully
        '''
        Young = 210000
        Poisson = 0.3
        problem = 0
        mat = material_routine(problem)
        actual_C = mat.planestress()
        expected_C = (Young/(1-Poisson**2))*np.array([[1,Poisson,0],[Poisson,1,0],[0,0,(1-Poisson)/2]])
        assert (array_equiv(actual_C,expected_C)) is True
        
    def test_plane_strain_true(self):
        '''
        UNIT TESTING
        Aim: Test plane strain stiffness tensor from planestrain method in material_routine class
    
        Expected result : Array of stiffness tensor of size (3,3)
    
        Test command : pytest test_material_routine.py::Test_material_routine
    
        Remarks : test case passed successfully
        '''
        Young = 210000
        Poisson = 0.3
        problem = 0
        mat = material_routine(problem)
        actual_C = mat.planestrain()
        expected_C = ((Young)*(1-Poisson))/((1-Poisson*2)*(1+Poisson))*np.array([[1,Poisson/(1-Poisson),0],[Poisson/(1-Poisson),1,0],[0,0,(1-2*Poisson)/(2*(1-Poisson))]])
        assert (array_equiv(actual_C,expected_C)) is True
    
    def test_material_routine_elastic_true(self):
        '''
        UNIT TESTING
        Aim: Test material routine for elastic case in material_routine class
    
        Expected result : Array of stiffness tensor for plane strain case
    
        Test command : pytest test_material_routine.py::Test_material_routine
    
        Remarks : test case passed successfully
        '''
        Young = 210000
        Poisson = 0.3
        problem = 0
        mat = material_routine(problem)
        actual_C = mat.material_elasticity()
        expected_C = ((Young)*(1-Poisson))/((1-Poisson*2)*(1+Poisson))*np.array([[1,Poisson/(1-Poisson),0],[Poisson/(1-Poisson),1,0],[0,0,(1-2*Poisson)/(2*(1-Poisson))]])
        assert (array_equiv(actual_C,expected_C)) is True
    
    def test_material_routine_elastic_plastic_true(self):
        '''
        UNIT TESTING
        Aim: Test material routine for elastic-plastic case giving elastic conditions 
        (strain, plastic strain and hardening equal to zero) in material_routine class
    
        Expected result : Array of stiffness tensor for plane strain case
    
        Test command : pytest test_material_routine.py::Test_material_routine
    
        Remarks : test case passed successfully
        '''
        # Shear modulus
        shear = 70300  # Mpa
        # Bulk modulus
        bulk = 136500  # Mpa
        problem = 1
        load_type = 0
        Young = (9 * bulk * shear) / ((3 * bulk) + shear)
        Poisson = ((3 * bulk) - (2 * shear)) / ((6 * bulk) + (2 * shear))
        strain = np.array([0,0,0])
        strain_plas = np.array([0,0,0])
        alpha = 0
        mat = material_routine(problem,load_type)
        a,actual_C,b,c = mat.material_plasticity(strain, strain_plas, alpha)
        expected_C = ((Young)*(1-Poisson))/((1-Poisson*2)*(1+Poisson))*np.array([[1,Poisson/(1-Poisson),0],[Poisson/(1-Poisson),1,0],[0,0,(1-2*Poisson)/(2*(1-Poisson))]])
        assert (array_equiv(actual_C,expected_C)) is True
        
# test = Test_material_routine()
# test.test_plane_stress_true()
# test.test_plane_strain_true()
# test.test_material_routine_elastic_plastic_true()
# test.test_material_routine_elastic_true()
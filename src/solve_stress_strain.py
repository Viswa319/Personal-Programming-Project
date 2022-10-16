# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
from shape_function import shape_function
from Bmatrix import Bmatrix
class solve_stress_strain():
    """Class to compute stress, strain and strain energy at integration points for each element.
    
    Strain energy is computed and compared to previous value and updated accordingly. 
    It is computed using elastic strain and stress.
    """
    def __init__(self,num_Gauss_2D, num_stress, num_node_elem, num_dof_u, elements, disp, Points, nodes, C, strain_energy, problem, strain_plas = None, alpha = None, hardening = None, sigma_y = None):
        """
        Class to compute stress, strain and strain energy at integration points for each element.

        Parameters
        ----------
        num_Gauss_2D : int
            Total number of Gauss points used for integration in 2-dimension.
        num_stress : int
            Number of independent stress components.
        num_node_elem : int
            Number of nodes per element.
        num_dof_u : int
            The number of DOFs for displacements per node.
        elements : Array of int, size(num_elem,num_node_elem)
            Element connectivity matrix.
        disp : Array of float64, size(num_tot_var_u)
            Displacements.
        Points : Array of float64, size(num_dim,num_Gauss**num_dim)
            Gauss points used for integration.
        nodes : Array of float64, size(num_node,num_dim)
            Co-ordinates of all field nodes.
        C : Array of float64, size(3,3)
            Stiffness tensor.
        Strain_energy : Array of float64, size(num_elem,num_Gauss_2D)
            Strain energy at the integration points for all elements.
        problem : int
            If problem = 0 elastic, \n
            if problem = 1 elastic-plastic brittle, \n
        strain_plas : Array of float64, size(num_elem,num_Gauss_2D,num_stress)
            Plastic strain at the integration points for all elements.
        alpha : float64
            Scalar hardening variable at previous step.
        hardening : float64
            Hardening modulus.
        sigma_y : float64
            Yield stress.

        Returns
        -------
        None.

        """
        self.num_elem = len(elements)
        self.num_Gauss_2D = num_Gauss_2D
        self.num_stress = num_stress
        self.num_node_elem = num_node_elem
        self.num_dof_u = num_dof_u
        self.elements = elements
        self.disp = disp
        self.Points = Points
        self.nodes = nodes
        self.C = C
        self.strain_energy = strain_energy
        self.problem = problem
        self.strain_plas = strain_plas
        self.alpha = alpha
        self.hardening = hardening
        self.sigma_y = sigma_y
        self.solve()
        
    def solve(self):
        """
        Function for solving stress, strain and strain energy at integration points for each element

        Returns
        -------
        None.

        """
        num_elem_var_u = self.num_node_elem*self.num_dof_u # total number of displacements per element 
        
        # Initialize stress component values at the integration points for all elements
        self.stress = np.zeros((self.num_elem,self.num_Gauss_2D,self.num_stress))
        
        # Initialize strain component values at the integration points for all elements
        self.strain = np.zeros((self.num_elem,self.num_Gauss_2D,self.num_stress))
        
        # Initialize strain component values at the integration points for all elements
        self.strain_elas = np.zeros((self.num_elem,self.num_Gauss_2D,self.num_stress))
        
        # Initialize strain energy at the integration points for all elements
        self.strain_energy_new = np.zeros((self.num_elem,self.num_Gauss_2D))
        
        # Initialize element displacement vector
        elem_disp = np.zeros((self.num_elem,num_elem_var_u))
        
        # Getting displacements belongs to respective elements
        for i in range(self.num_node_elem):
            elem_node = self.elements[:,i]
            for j in range(self.num_dof_u):
                i_elem_var = i*self.num_dof_u + j
                i_tot_var = (elem_node-1)*self.num_dof_u + j
                elem_disp[:,i_elem_var] = self.disp[i_tot_var]
        
        # loop for all elements
        for elem in range(self.num_elem):
            # Gettting co-ordinates of respective element
            elem_node_1 = self.elements[elem,:]
            elem_coord = self.nodes[elem_node_1-1,:]
            
            # loop for all Gauss points
            for j in range(self.num_Gauss_2D):
                gpos = self.Points[j]
                
                # Calling shape function and its derivatives from shape function class
                shape = shape_function(self.num_node_elem,gpos,elem_coord)
                dNdX = shape.get_shape_function_derivative()
                
                # Compute B matrix
                B = Bmatrix(dNdX,self.num_node_elem)
                Bmat = B.Bmatrix_disp()
                
                
                # Compute strain values at the integration points for all elements
                self.strain[elem,j,:] = np.matmul(Bmat,elem_disp[elem])
                
                if self.problem == 0:
                    # Compute stress values at the integration points for all elements
                    self.stress[elem,j,:] = np.matmul(self.C,self.strain[elem,j,:])
                    # Compute strain energy values at the integration points for all elements
                    self.strain_energy_new[elem,j] = 0.5*np.dot(self.stress[elem,j,:],self.strain[elem,j,:])
                elif self.problem == 1 or self.problem == 2:
                    # Compute elastic strain at the integration points for all elements
                    self.strain_elas[elem,j,:] = self.strain[elem,j,:] - self.strain_plas[elem,j,:]
                
                    # Compute stress values at the integration points for all elements
                    self.stress[elem,j,:] = np.matmul(self.C,self.strain_elas[elem,j,:])
                
                    strain_energy_plas = ((1 / 2) * self.hardening * (self.alpha[elem]**2)) + (self.sigma_y * self.alpha[elem])
                    # Compute strain energy values at the integration points for all elements
                    self.strain_energy_new[elem,j] = 0.5*np.dot(self.stress[elem,j,:],self.strain_elas[elem,j,:]) + strain_energy_plas
                
                # Checks strain energy value with previous strain energy value and updates accordingly
                if self.strain_energy_new[elem,j] > self.strain_energy[elem,j]:
                    self.strain_energy[elem,j] = self.strain_energy_new[elem,j]
                else:
                    self.strain_energy[elem,j] = self.strain_energy[elem,j]
    
    @property
    def solve_stress(self):
        """
        Calls solve class and returns stress 

        Returns
        -------
        Array of float64, size(num_elem,num_Gauss_2D,num_stress)
            Stress at the integration points for all elements.

        """
        return self.stress
    
    @property
    def solve_strain(self):
        """
        Calls solve class and returns strain 

        Returns
        -------
        Array of float64, size(num_elem,num_Gauss_2D,num_stress)
            Strain at the integration points for all elements.

        """
        return self.strain
    
    @property
    def solve_strain_energy(self):
        """
        Calls solve class and returns strain energy 

        Returns
        -------
        Array of float64, size(num_elem,num_Gauss_2D)
            Strain energy at the integration points for all elements.

        """
        return self.strain_energy
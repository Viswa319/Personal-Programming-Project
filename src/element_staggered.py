# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
from shape_function import shape_function
from Bmatrix import Bmatrix
from material_routine import material_routine

class element_staggered():
    """Class to compute element stiffness tensor for displacments and phase field order parameter,
                        element residual for phase field order parameter,
                        element internal force for displacements.
       
        This element routine is computed based on staggered scheme.                 
    """
    def __init__(self, elem, Points, Weights, disp, Phi, stress, strain_energy, elements, nodes, num_dof, num_node_elem, num_Gauss_2D):
        """
        Class to compute element stiffness tensor for displacments and phase field order parameter
                        element residual for phase field order parameter
                        element internal force for displacements

        Parameters
        ----------
        elem : int
            DESCRIPTION.
        Points : Array of float64, size(num_dim,num_Gauss**num_dim)
            Gauss points used for integration.
        Weights : Array of float64, size(num_Gauss**num_dim)
            Weights for Gauss points used for integration.
        disp : Array of float64, size(num_tot_var_u)
            Displacements.
        Phi : Array of float64, size(num_tot_var_phi)
            Phase field order parameter.
        stress : Array of float64, size(num_elem,num_Gauss_2D,num_stress)
            Stress at the integration points for all elements.
        strain : Array of float64, size(num_elem,num_Gauss_2D,num_stress)
            Strain at the integration points for all elements.
        strain_energy : Array of float64, size(num_elem,num_Gauss_2D)
            Strain energy at the integration points for all elements.
        elements : Array of int, size(num_elem,num_node_elem)
            Element connectivity matrix.
        nodes : Array of float64, size(num_node,num_dim)
            Co-ordinates of all field nodes.
        num_dof : int
            Total number of DOFs per node.
        num_node_elem : int
            Number of nodes per element.
        num_Gauss_2D : int
            Total number of Gauss points used for integration in 2-dimension.

        Returns
        -------
        None.

        """
        self.elem = elem
        self.Points = Points
        self.Weights = Weights
        self.disp = disp
        self.Phi = Phi
        self.stress = stress
        self.strain_energy = strain_energy
        self.elem_node = elements[self.elem, :]
        self.elem_coord = nodes[self.elem_node - 1, :]
        self.elements = elements
        self.nodes = nodes
        self.num_dof = num_dof
        self.num_node_elem = num_node_elem
        self.num_Gauss_2D = num_Gauss_2D

    def element_stiffness_displacement(self, C:np.array, k_const:float):
        """
        Function for calculating element stiffness matrix and internal force vector 
        for displacement for elastic case.

        Parameters
        ----------
        C : Array of float64, size(k)
            Stiffness tensor.
        k_const : float64
            parameter to avoid overflow for cracked elements.

        Returns
        -------
        K_uu : Array of float64, size(num_elem_var_u,num_elem_var_u)
            element stiffness matrix.
        
        F_int_elem : Array of float64, size(num_elem_var_u)
            element internal force vector.
        
        """
        num_dof_u = self.num_dof - 1

        # total number of displacements per element
        num_elem_var_u = self.num_node_elem * num_dof_u

        # Initialize element stiffness matrix
        K_uu = np.zeros((num_elem_var_u, num_elem_var_u))

        # Initialize element internal force vector
        F_int_elem = np.zeros(num_elem_var_u)

        i_tot_var = self.elem_node
        elem_phi = self.Phi[i_tot_var - 1]

        for j in range(0, self.num_Gauss_2D):
            gpos = self.Points[j]

            # Calling shape function and its derivatives from shape function class
            shape = shape_function(self.num_node_elem, gpos, self.elem_coord)
            N = shape.get_shape_function()
            dNdX = shape.get_shape_function_derivative()
            # Call detereminant of Jacobian
            det_Jacobian = shape.get_det_Jacobian()

            # phase field order parameter
            phi = np.matmul(N[0], elem_phi)
            if phi > 1:
                phi = 1

            # Compute B matrix
            B = Bmatrix(dNdX, self.num_node_elem)
            Bmat = B.Bmatrix_disp()

            # Compute element stiffness matrix
            K_uu = K_uu + np.matmul(np.matmul(np.transpose(Bmat), C), Bmat) * (((1 - phi) ** 2) + k_const) \
                * self.Weights[j] * det_Jacobian
            
            # Compute internal force vector for an element
            F_int_elem = F_int_elem + np.matmul(np.transpose(Bmat), self.stress[self.elem, j, :]) * (((1 - phi) ** 2) + k_const) \
                * self.Weights[j] * det_Jacobian
        return K_uu, F_int_elem

    def element_stiffness_field_parameter(self, G_c, l_0):
        """
        Function for calculating element stiffness matrix and residual vector 
        for phase field order parameter (phi).

        Parameters
        ----------
        G_c : float64
            critical energy release for unstable crack or damage.
        l_0 : float64
            length parameter which controls the spread of damage.

        Returns
        -------
        K_phiphi : Array of float64, size(num_elem_var_phi,num_elem_var_phi)
            element stiffness matrix for phase-field order parameter.
        
        residual_phi : Array of float64, size(num_elem_var_phi)
            element residual vector for phase-field order parameter.
            
        """

        # the number of DOFs for order parameters per node
        num_dof_phi = self.num_dof - 2

        # total number of order parameters per element
        num_elem_var_phi = self.num_node_elem * num_dof_phi

        # Initialize element stiffness matrix
        K_phiphi = np.zeros((num_elem_var_phi, num_elem_var_phi))

        # Initialize residual vector for order parameter
        residual_phi = np.zeros(num_elem_var_phi)

        i_tot_var = self.elem_node
        elem_phi = self.Phi[i_tot_var - 1]

        for j in range(0, self.num_Gauss_2D):
            gpos = self.Points[j]

            # Calling shape function and its derivatives from shape function class
            shape = shape_function(self.num_node_elem, gpos, self.elem_coord)
            N = shape.get_shape_function()
            dNdX = shape.get_shape_function_derivative()
            # Call detereminant of Jacobian
            det_Jacobian = shape.get_det_Jacobian()

            # Compute B matrix
            B = Bmatrix(dNdX, self.num_node_elem)
            Bmat = B.Bmatrix_phase_field()

            # phase field order parameter
            phi = np.matmul(N[0], elem_phi)
            if phi > 1:
                phi = 1

            # Strain energy
            H = self.strain_energy[self.elem, j]

            # Compute element stiffness matrix
            K_phiphi = K_phiphi + (G_c * l_0 * np.matmul(np.transpose(Bmat), Bmat) +
                + ((G_c / l_0) + 2 * H) * np.matmul(np.transpose([N[0]]), [N[0]]))\
                * self.Weights[j] * det_Jacobian

            # Compute residual vector for order parameter
            residual_phi = residual_phi + (-2*(1-phi)*(N[0]*H) + \
                + ((G_c/l_0)*N[0]*phi)+ (G_c*l_0)*np.matmul(np.transpose(Bmat),np.matmul(Bmat,elem_phi)))\
                *self.Weights[j]*det_Jacobian

        return K_phiphi, residual_phi
    
    def element_residual_field_parameter(self,G_c,l_0):
        """
        Function for calculating residual vector for order parameter

        Parameters
        ----------
        G_c : float64
            critical energy release for unstable crack or damage.
        l_0 : float64
            length parameter which controls the spread of damage.

        Returns
        -------
        residual_phi : Array of float64, size(num_elem_var_phi)
            residual vector for order parameter.

        """
        # the number of DOFs for order parameters per node        
        num_dof_phi = self.num_dof-2
        
        # total number of order parameters per element 
        num_elem_var_phi = self.num_node_elem*num_dof_phi
    
        # Initialize residual vector for order parameter
        residual_phi = np.zeros(num_elem_var_phi)
        
        i_tot_var = self.elem_node
        elem_phi = self.Phi[i_tot_var-1]
        
        for j in range(0,self.num_Gauss_2D):
            gpos = self.Points[j]
            
            # Calling shape function and its derivatives from shape function class
            shape = shape_function(self.num_node_elem,gpos,self.elem_coord)
            N = shape.get_shape_function()
            dNdX = shape.get_shape_function_derivative()
            # Call detereminant of Jacobian
            det_Jacobian = shape.get_det_Jacobian()
            
            # phase field order parameter
            phi = np.matmul(N[0],elem_phi)
            if phi > 1:
                phi = 1 
            
            # Compute B matrix
            B = Bmatrix(dNdX,self.num_node_elem)
            Bmat = B.Bmatrix_phase_field()
            
            # Strain energy
            H = self.strain_energy[self.elem,j]
            
            # Compute residual vector for order parameter
            residual_phi = residual_phi + (-2*(1-phi)*(N[0]*H) + \
                + ((G_c/l_0)*N[0]*phi)+ (G_c*l_0)*np.matmul(np.transpose(Bmat),np.matmul(Bmat,elem_phi)))\
                *self.Weights[j]*det_Jacobian
        return residual_phi
    
    def element_stiffness_displacement_plasticity(self, k_const:float, strain:np.array, strain_plas:np.array, alpha:float):
        """
        Function for calculating element stiffness matrix and internal force vector 
        for displacement for elastic-plastic case and updates stress, plastic strain 
        and hardening variable from material routine.

        Parameters
        ----------
        k_const : float64
            Parameter to avoid overflow for cracked elements.
        
        strain : Array of float64, size(num_elem,num_Gauss_2D,num_stress)
            Strain at the integration points for all elements.
            
        strain_plas : Array of float64, size(num_elem,num_Gauss_2D,num_stress)
            Plastic strain at the previous step at integration points for all elements.
            
        alpha : float64
            Scalar hardening variable at previous step.
        Returns
        -------
        K_uu : Array of float64, size(num_elem_var_u,num_elem_var_u)
            element stiffness matrix.
        
        F_int_elem : Array of float64, size(num_elem_var_u)
            element internal force vector.
        
        stress : Array of float64, size(num_elem,num_Gauss_2D,num_stress)
            Stress at the integration points for all elements.
            
        strain_plas : Array of float64, size(num_elem,num_Gauss_2D,num_stress)
            Plastic strain at current step at integration points for all elements.
            
        alpha : float64
            Scalar hardening variable at current step.
            
        """
        problem = 'Elastic-Plastic'
        num_dof_u = self.num_dof - 1

        # total number of displacements per element
        num_elem_var_u = self.num_node_elem * num_dof_u

        # Initialize element stiffness matrix
        K_uu = np.zeros((num_elem_var_u, num_elem_var_u))

        # Initialize element internal force vector
        F_int_elem = np.zeros(num_elem_var_u)

        i_tot_var = self.elem_node
        elem_phi = self.Phi[i_tot_var - 1]

        for j in range(0, self.num_Gauss_2D):
            material = material_routine(problem)
            stress_new, C, strain_plas_new, alpha_new = material.material_plasticity(strain[self.elem,j,:],strain_plas[self.elem,j,:],alpha)
            gpos = self.Points[j]

            # Calling shape function and its derivatives from shape function class
            shape = shape_function(self.num_node_elem, gpos, self.elem_coord)
            N = shape.get_shape_function()
            dNdX = shape.get_shape_function_derivative()
            # Call detereminant of Jacobian
            det_Jacobian = shape.get_det_Jacobian()

            # phase field order parameter
            phi = np.matmul(N[0], elem_phi)
            if phi > 1:
                phi = 1

            # Compute B matrix
            B = Bmatrix(dNdX, self.num_node_elem)
            Bmat = B.Bmatrix_disp()
            
            # Compute element stiffness matrix
            K_uu = K_uu + np.matmul(np.matmul(np.transpose(Bmat), C), Bmat) * (((1 - phi) ** 2) + k_const) \
                * self.Weights[j] * det_Jacobian
            
            # Compute internal force vector for an element
            F_int_elem = F_int_elem + np.matmul(np.transpose(Bmat), stress_new) * (((1 - phi) ** 2) + k_const) \
                * self.Weights[j] * det_Jacobian
            
            # Update stress, plastic strain and hardening variable
            self.stress[self.elem,j,:] = stress_new
            strain_plas[self.elem,j,:] = strain_plas_new
            alpha = alpha_new
        return K_uu, F_int_elem, self.stress, strain_plas, alpha
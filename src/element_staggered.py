# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
from shape_function import shape_function
from Bmatrix import Bmatrix
from input_abaqus import *
class element_staggered():
    def __init__(self,elem,Points,Weights,C,disp,Phi,stress,strain,strain_energy):
        self.elem = elem
        self.Points = Points
        self.Weights = Weights
        self.C = C
        self.disp = disp
        self.Phi = Phi
        self.stress = stress
        self.strain = strain
        self.strain_energy = strain_energy
        self.elem_node = elements[self.elem,:]
        self.elem_coord = nodes[self.elem_node-1,:]
        self.strain_energy_new = np.zeros((num_elem,num_Gauss**num_dim))
    def element_stiffness_displacement(self):
        """
        Function for calculating element stiffness matrix.
    
        Parameters
        ----------
        elem : int
            element number.
        Points :  Array of float64, size(num_dim,num_Gauss**num_dim)
            Gauss points used for integration.
        Weights : Array of float64, size(num_Gauss**num_dim)
            weights used for integration.
        C : Array of float64, size(3,3)
            Stiffness tensor.
    
        Raises
        ------
        ValueError
            Solution will be terminated if determinant of Jacobian is either zero or negative.
    
        Returns
        -------
        K_uu : Array of float64, size(num_elem_var,num_elem_var)
            element stiffness matrix.
    
        """
        num_dof_u = num_dof-1
        
        # total number of variables for displacements
        num_tot_var_u = num_node*num_dof_u
        
        # total number of displacements per element 
        num_elem_var_u = num_node_elem*num_dof_u
        
        # Initialize element stiffness matrix   
        K_uu = np.zeros((num_elem_var_u,num_elem_var_u))
        
        i_tot_var = self.elem_node
        elem_phi = self.Phi[i_tot_var-1]
        

        for j in range(0,num_Gauss**num_dim):
            gpos = self.Points[j]
            
            shape = shape_function(num_node_elem,gpos,self.elem_coord)
            N = shape.get_shape_function()
            dNdX = shape.get_shape_function_derivative()
            det_Jacobian = shape.get_det_Jacobian()
            phi = np.matmul(N[0],elem_phi)
            
            # Compute B matrix
            B = Bmatrix(dNdX,num_node_elem)
            Bmat = B.Bmatrix_disp()
            
            K_uu = K_uu + np.matmul(np.matmul(np.transpose(Bmat),self.C),Bmat)*(((1-phi)**2)+k_const) \
                *self.Weights[j]*det_Jacobian
                
        return K_uu
    def element_stiffness_field_parameter(self):
        """
        Function for calculating element stiffness matrix.
    
        Parameters
        ----------
        elem : int
            element number.
        Points :  Array of float64, size(num_dim,num_Gauss**num_dim)
            Gauss points used for integration.
        Weights : Array of float64, size(num_Gauss**num_dim)
            weights used for integration.
        C : Array of float64, size(3,3)
            Stiffness tensor.
    
        Raises
        ------
        ValueError
            Solution will be terminated if determinant of Jacobian is either zero or negative.
    
        Returns
        -------
        K_phiphi : Array of float64, size(num_elem_var,num_elem_var)
            element stiffness matrix.
    
        """
        num_dof_phi = num_dof-2
        
        # total number of variables for displacements
        num_tot_var_phi = num_node*num_dof_phi
        
        # the number of DOFs for order parameters per node
        num_dof_phi = num_dof-2
        
        # total number of order parameters per element 
        num_elem_var_phi = num_node_elem*num_dof_phi
        
        # Initialize element stiffness matrix
        K_phiphi = np.zeros((num_elem_var_phi,num_elem_var_phi))
        
        # Initialize element stiffness matrix
        residual_phi = np.zeros(num_elem_var_phi)
        
        i_tot_var = self.elem_node
        elem_phi = self.Phi[i_tot_var-1]
        
        for j in range(0,num_Gauss**num_dim):
            gpos = self.Points[j]
            
            shape = shape_function(num_node_elem,gpos,self.elem_coord)
            N = shape.get_shape_function()
            dNdX = shape.get_shape_function_derivative()
            det_Jacobian = shape.get_det_Jacobian()
            phi = np.matmul(N[0],elem_phi)
            
            # Compute B matrix
            B = Bmatrix(dNdX,num_node_elem)
            Bmat = B.Bmatrix_phase_field()
            self.strain_energy_new[self.elem,j] = 0.5*np.dot(self.stress[self.elem,j,:],self.strain[self.elem,j,:])
            
            if self.strain_energy_new[self.elem,j] > self.strain_energy[self.elem,j]:
                H = self.strain_energy_new[self.elem,j]
            else:
                H = self.strain_energy[self.elem,j]
            K_phiphi = K_phiphi + (G_c*l_0*np.matmul(np.transpose(Bmat),Bmat) + \
                + ((G_c/l_0)+2*H)*np.matmul(np.transpose([N[0]]),[N[0]]))\
                *self.Weights[j]*det_Jacobian
            
            residual_phi = residual_phi + (-2*(1-phi)*(N[0]*H) + \
                + ((G_c/l_0)*N[0]*phi)+ (G_c*l_0)*np.matmul(np.transpose(Bmat),np.matmul(Bmat,elem_phi)))\
                *self.Weights[j]*det_Jacobian
            
            self.strain_energy[self.elem,j] = self.strain_energy_new[self.elem,j]
        return K_phiphi,residual_phi,self.strain_energy
    
    def element_internal_force(self):
        """
        Function for calculating element stiffness matrix.
    
        Parameters
        ----------
        elem : int
            element number.
        Points :  Array of float64, size(num_dim,num_Gauss**num_dim)
            Gauss points used for integration.
        Weights : Array of float64, size(num_Gauss**num_dim)
            weights used for integration.
        C : Array of float64, size(3,3)
            Stiffness tensor.
    
        Raises
        ------
        ValueError
            Solution will be terminated if determinant of Jacobian is either zero or negative.
    
        Returns
        -------
        K_uu : Array of float64, size(num_elem_var,num_elem_var)
            element stiffness matrix.
    
        """
        num_dof_u = num_dof-1
                
        # total number of displacements per element 
        num_elem_var_u = num_node_elem*num_dof_u
        
        # Initialize element stiffness matrix   
        F_int_elem = np.zeros(num_elem_var_u)
        
        i_tot_var = self.elem_node
        elem_phi = self.Phi[i_tot_var-1]
        

        for j in range(0,num_Gauss**num_dim):
            gpos = self.Points[j]
            
            shape = shape_function(num_node_elem,gpos,self.elem_coord)
            N = shape.get_shape_function()
            dNdX = shape.get_shape_function_derivative()
            det_Jacobian = shape.get_det_Jacobian()
            phi = np.matmul(N[0],elem_phi)
            
            # Compute B matrix
            B = Bmatrix(dNdX,num_node_elem)
            Bmat = B.Bmatrix_disp()
            
            F_int_elem = F_int_elem + np.matmul(np.transpose(Bmat),self.stress[self.elem,j,:])*(((1-phi)**2)+k_const) \
                *self.Weights[j]*det_Jacobian
            
        return F_int_elem
    
    def element_residual_field_parameter(self):
        """
        Function for calculating element stiffness matrix.
    
        Parameters
        ----------
        elem : int
            element number.
        Points :  Array of float64, size(num_dim,num_Gauss**num_dim)
            Gauss points used for integration.
        Weights : Array of float64, size(num_Gauss**num_dim)
            weights used for integration.
        C : Array of float64, size(3,3)
            Stiffness tensor.
    
        Raises
        ------
        ValueError
            Solution will be terminated if determinant of Jacobian is either zero or negative.
    
        Returns
        -------
        K_phiphi : Array of float64, size(num_elem_var,num_elem_var)
            element stiffness matrix.
    
        """
        num_dof_phi = num_dof-2
        
        # total number of variables for displacements
        num_tot_var_phi = num_node*num_dof_phi
        
        # the number of DOFs for order parameters per node
        num_dof_phi = num_dof-2
        
        # total number of order parameters per element 
        num_elem_var_phi = num_node_elem*num_dof_phi
    
        # Initialize element stiffness matrix
        residual_phi = np.zeros(num_elem_var_phi)
        
        i_tot_var = self.elem_node
        elem_phi = self.Phi[i_tot_var-1]
        
        for j in range(0,num_Gauss**num_dim):
            gpos = self.Points[j]
            
            shape = shape_function(num_node_elem,gpos,self.elem_coord)
            N = shape.get_shape_function()
            dNdX = shape.get_shape_function_derivative()
            det_Jacobian = shape.get_det_Jacobian()
            phi = np.matmul(N[0],elem_phi)
            
            # Compute B matrix
            B = Bmatrix(dNdX,num_node_elem)
            Bmat = B.Bmatrix_phase_field()
            self.strain_energy_new[self.elem,j] = 0.5*np.dot(self.stress[self.elem,j,:],self.strain[self.elem,j,:])
            
            if self.strain_energy_new[self.elem,j] > self.strain_energy[self.elem,j]:
                H = self.strain_energy_new[self.elem,j]
            else:
                H = self.strain_energy[self.elem,j]
            
            residual_phi = residual_phi + (-2*(1-phi)*(N[0]*H) + \
                + ((G_c/l_0)*N[0]*phi)+ (G_c*l_0)*np.matmul(np.transpose(Bmat),np.matmul(Bmat,elem_phi)))\
                *self.Weights[j]*det_Jacobian
            self.strain_energy[self.elem,j] = self.strain_energy_new[self.elem,j]
        return residual_phi,self.strain_energy
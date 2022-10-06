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
    def __init__(self,num_elem,num_Gauss_2D,num_stress,num_node_elem,num_dof_u,elements,disp,Points,nodes,C,Strain_energy):
        self.num_elem = num_elem
        self.num_Gauss_2D = num_Gauss_2D
        self.num_stress = num_stress
        self.num_node_elem = num_node_elem
        self.num_dof_u = num_dof_u
        self.elements = elements
        self.disp = disp
        self.Points = Points
        self.nodes = nodes
        self.C = C
        self.Strain_energy = Strain_energy
        self.solve()
        
    def solve(self):
        num_elem_var_u = self.num_node_elem*self.num_dof_u # total number of displacements per element 
        self.stress = np.zeros((self.num_elem,self.num_Gauss_2D,self.num_stress))
        self.strain = np.zeros((self.num_elem,self.num_Gauss_2D,self.num_stress))
        self.strain_energy_new = np.zeros((self.num_elem,self.num_Gauss_2D))
        
        elem_disp = np.zeros((self.num_elem,num_elem_var_u))
        
        for i in range(self.num_node_elem):
            elem_node = self.elements[:,i]
            for j in range(self.num_dof_u):
                i_elem_var = i*self.num_dof_u + j
                i_tot_var = (elem_node-1)*self.num_dof_u + j
                elem_disp[:,i_elem_var] = self.disp[i_tot_var]
        for elem in range(self.num_elem):
            elem_node_1 = self.elements[elem,:]
            elem_coord = self.nodes[elem_node_1-1,:]
            for j in range(self.num_Gauss_2D):
                gpos = self.Points[j]
                   
                shape = shape_function(self.num_node_elem,gpos,elem_coord)
                dNdX = shape.get_shape_function_derivative()
                
                # Compute B matrix
                B = Bmatrix(dNdX,self.num_node_elem)
                Bmat = B.Bmatrix_disp()
                   
                self.strain[elem,j,:] = np.matmul(Bmat,elem_disp[elem])#self.strain[elem,j,:] + np.matmul(Bmat,elem_disp[elem])
                   
                self.stress[elem,j,:] = np.matmul(self.C,self.strain[elem,j,:])#self.stress[elem,j,:] + np.matmul(self.C,self.strain[elem,j,:])
                
                self.strain_energy_new[elem,j] = 0.5*np.dot(self.stress[elem,j,:],self.strain[elem,j,:])
                
                if self.strain_energy_new[elem,j] > self.Strain_energy[elem,j]:
                    self.Strain_energy[elem,j] = self.strain_energy_new[elem,j]
                else:
                    self.Strain_energy[elem,j] = self.Strain_energy[elem,j]
    @property
    def solve_stress(self):
        return self.stress
    @property
    def solve_strain(self):
        return self.strain
    @property
    def solve_strain_energy(self):
        return self.Strain_energy
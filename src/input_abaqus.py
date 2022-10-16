# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np

class input_parameters():
    """Input parameters class where all inputs are defined. 
    Four different functions are implemented based on the input type.
    """
    def __init__(self,load_type, problem):
        self.load_type = load_type
        self.problem = problem
    
    def geometry_parameters(self):
        """
        Input parameters related to geometry are defined.
        **Nodes** and **elements** of a geometry are generated in ABAQUS.
        ABAQUS input file has been modified such that same program is valid for all files.
        
        Returns
        -------
        num_dim : int
            Dimension of a problem.
        num_node_elem : int
            Number of nodes per element.
        nodes : Array of float64, size(num_node,num_dim)
            Co-ordinates of all field nodes.
        num_node : int
            Number of nodes.
        elements : Array of int, size(num_elem,num_node_elem)
            Element connectivity matrix.
        num_elem : int
            Total number of elements.
        num_dof : int
            Total number of DOFs per node.
        num_Gauss : int
            Total number of Gauss points used for integration in 1-dimension.
        num_stress : int
            Number of independent stress components.
        problem : str
            Problem type either 'Elastic' or 'Elastic_Plastic'.

        """
        # dimension of a problem
        num_dim = 2
        
        # number of nodes per element
        num_node_elem = 4
        
        if self.load_type == 0:
            file_name = 'tensile.inp'
        elif self.load_type == 1:
            # file_name = 'shear_coarse.inp'
            file_name = 'shear_fine.inp'
        
        with open(file_name, "r") as f:
        
            lines = f.readlines()
            # number of nodes
            for i in range(len(lines)):
                if lines[i] == "*Nset, nset=Set-1, generate\n":
                    num_node = int((lines[i + 1].split(',')[1]))
                
                if lines[i] == "*Node\n":
                    start_node = i + 1
                
                if lines[i] == "*Elset, elset=Set-1, generate\n":
                    num_elem = int((lines[i + 1].split(',')[1]))
                    break
                
                if lines[i] == "*Element\n":
                    start_elem = i + 1
                
        
            # extracting nodes from input file
            nodes = np.zeros((num_node, num_dim), float)
            for i in range(start_node, start_node + num_node):
                nodes[i - start_node, 0] = (lines[i].split(','))[1]
                nodes[i - start_node, 1] = (lines[i].split(','))[2]
        
        
            # extracting elements and material type of each element from input file
            elements = np.zeros((num_elem, num_node_elem), int)
            for i in range(start_elem, start_elem + num_elem):
                elements[i - start_elem, 0:num_node_elem] = (lines[i].split(','))[1:num_node_elem + 1]
        
            del start_elem, start_node
        
        # Data used for one elemnet test, while executing one element test uncomment these lines.
        # nodes = np.array([[0,0],[1,0],[1,1],[0,1]])
        # num_node = len(nodes)
        # elements = np.array([[1,2,3,4]])
        # num_elem = len(elements)

        
        # total number of degrees of freedom ----> 2 displacement and 1 phase field
        num_dof = 3
        
        # number of Gauss points for integration
        num_Gauss = 2
        
        # number of independent stress components
        num_stress = 3
        
        return num_dim, num_node_elem, nodes,num_node,elements,num_elem, num_dof, num_Gauss, num_stress
    
    def material_parameters_elastic(self):
        """
        Material specific input parameters are defined.
        This function is for **elastic** problem.
        
        Returns
        -------
        k_const : float64
            Parameter to avoid overflow for cracked elements.
        G_c : float64
            Critical energy release for unstable crack or damage.
        l_0 : float64
            Length parameter which controls the spread of damage.
        Young : float64
            Youngs modulus.
        Poisson : float64
            Poissons ratio.
        stressState : int
            If stressState == 1 plane stress; if stressState = 2 plane strain.

        """
        # Material specific parameters
        
        # parameter to avoid overflow for cracked elements
        k_const = 1e-7
        
        # critical energy release for unstable crack or damage
        G_c = 2.7  # MPa mm
        
        # length parameter which controls the spread of damage
        l_0 = 0.04  # mm
        
        # Young's modulus
        Young = 210000  # Mpa
        
        # Poisson's ratio
        Poisson = 0.3
        
        # if stressState == 1 plane stress; if stressState = 2 plane strain
        stressState = 2
        
        return k_const,G_c,l_0,Young,Poisson,stressState
        
    def material_parameters_elastic_plastic(self):
        """

        Material specific input parameters are defined.
        This function is for **elastic-plastic** problem.
        
        Returns
        -------
        k_const : float64
            Parameter to avoid overflow for cracked elements.
        G_c : float64
            Critical energy release for unstable crack or damage.
        l_0 : float64
            Length parameter which controls the spread of damage.
        stressState : int
            If stressState == 1 plane stress; if stressState = 2 plane strain.
        shear : float64
            Shear modulus.
        bulk : float64
            Bulk modulus.
        sigma_y : float64
            Yield stress.
        hardening : float64
            Hardening modulus.

        """
        # Material specific parameters
        
        # parameter to avoid overflow for cracked elements
        k_const = 1e-7
        
        # critical energy release for unstable crack or damage
        G_c = 20.9  # MPa mm
        
        # length parameter which controls the spread of damage
        l_0 = 0.05  # mm
        
        # Shear modulus
        shear = 70300  # Mpa
        
        # Bulk modulus
        bulk = 136500  # Mpa
        
        # Yield stress
        sigma_y = 443  # Mpa
        
        # Hardening modulus 
        hardening = 300  # Mpa
        
        # if stressState == 1 plane stress; if stressState = 2 plane strain
        stressState = 2
        
        return k_const, G_c, l_0, stressState, shear, bulk, sigma_y, hardening

    def time_integration_parameters(self):
        """
        Inputs for time integration parameters are defined.

        Returns
        -------
        num_step : int
            Number of time steps.
        max_iter : int
            Maximum number of Newton-Raphson iterations.
        max_tol : float64
            Tolerance for iterative solution.
        disp_inc : float64
            Displacmenet increment per time steps.

        """
        
        # Inputs for time integration parameters
        
        # number of time steps
        num_step = 1000
        
        # maximum number of Newton-Raphson iterations
        max_iter = 10
        
        # tolerance for iterative solution
        max_tol = 1e-4
        
        # displacmenet increment per time steps
        disp_inc = 1e-5  # mm
        
        return num_step, max_iter, max_tol, disp_inc
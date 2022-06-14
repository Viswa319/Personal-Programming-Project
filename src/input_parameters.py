# *************************************************************************
#                 
#                      Upadhyayula Sai Viswanadha Sastry
#                               65130
# *************************************************************************
class Inputs:
    def __init__(self):
        # dimension of a problem
        self.dimension = 2 
        
        # Material properties
        self.Young = 3e7 # Young's modulus in N/m^2
        self.Poisson = 0.3 # Poisson ratio
        self.stressState = 1 # if plane stress 1, if plane strain 2
        
        # dimensions of domain ---> simple 2D rectangular beam
        self.xlength = 48 # Length in m
        self.ylength = 12 # Height in m
        
        # loading
        self.P = -1000 # Applied load in N
        
        # number of nodes in x- and y- axis respectively 
        self.ndivx = 25 # number of nodes in x-axis
        self.ndivy = 7 # number of nodes in y-axis
        
        # number of background points in x- and y- axis respectively 
        self.ndivxq = 25 # number of background points in x-axis
        self.ndivyq = 7 # number of background points in y-axis
        
        # if form == 0, cubic spline weight function is considered
        # if form == 1, quartic spline weight function is considered
        self.form = 0 # if form == 0, cubic spline weight function is considered
        
# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
import numpy.polynomial.legendre as ptwt
from geometry import quadrature_coefficients
from material_routine import material_routine 
with open('mesh_1.inp', "r") as f:
        lines = f.readlines()

        numnode = int(lines[0].split()[0]) # number of nodes
        numelem = int(lines[0].split()[1]) # number of elements
        numfixnodes = int(lines[0].split()[2]) # number of fixed nodes
        stressState = int(lines[0].split()[3]) # if stressState == 1 plane stress; if stressState = 2 plane strain
        numnode_elem = int(lines[0].split()[4]) # number of nodes per element
        numdof = int(lines[0].split()[5]) # degrees of freedom
        numdim = int(lines[0].split()[6]) # dimension of a problem
        numgauss = int(lines[0].split()[7]) # number of integration points
        numstress = int(lines[0].split()[8]) # number of stress comonents
        nummat = int(lines[0].split()[9]) # number of materials 
        numprops = int(lines[0].split()[10]) # number of material properties
        
        elements = np.zeros((numelem,numnode_elem),int)
        matno = np.zeros(numelem)
        for i in range(1,numelem+1):
            elements[i-1,0:numnode_elem] = (lines[i].split())[1:numnode_elem+1]
            matno[i-1] = (lines[i].split())[numnode_elem+1]
        nodes = np.zeros((numnode,numdim),float)
        for i in range(1+numelem,1+numelem+numnode):
            nodes[i-1-numelem,0] = float((lines[i].split())[1])
            nodes[i-1-numelem,1] = float((lines[i].split())[2])
        
        fixnodes = np.zeros(numfixnodes,int)
        fixdof = np.zeros((numfixnodes,numdof),int)
        disp_bc = np.zeros((numfixnodes,numdof),float)
        for i in range(1+numelem+numnode,1+numelem+numnode+numfixnodes):
            fixnodes[i-1-numelem-numnode] = (lines[i].split())[0]
            fixdof[i-1-numelem-numnode,0:numdof] = (lines[i].split())[1:numdof+1]
            disp_bc[i-1-numelem-numnode,0:numdof] = (lines[i].split())[3:numdof+3]
        
        props = np.zeros(numprops,float)
        props[0:numprops] = (lines[1+numelem+numnode+numfixnodes].split())[1:numprops+1]
        
        ipload = int((lines[1+numelem+numnode+numfixnodes+1].split())[0])
        nedge = int((lines[1+numelem+numnode+numfixnodes+1].split())[1])


# Call Guass quadrature points and weights using inbuilt function
PtsWts = ptwt.leggauss(numgauss)
points = PtsWts[0]
weights = PtsWts[1]
Points,Weights = quadrature_coefficients(numgauss)

# Call Stiffness tensor from material routine
mat = material_routine()
Young = props[0]
Poisson = props[1]
if stressState == 1:    
    C = mat.planestress(Young,Poisson)
elif stressState == 2:
    C = mat.planestrain(Young,Poisson)

# Initialization of force and global stiffness tensor
ak = np.zeros((numdof*numnode, numdof*numnode))
force = np.zeros(numdof*numnode)

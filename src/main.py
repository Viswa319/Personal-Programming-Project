# Sai Viswanadha Sastry Upadhyayula
# 65130
# Personal Programming Project
# Implementation of phase-field model of ductile fracture
import numpy as np
with open('mesh_1.inp', "r") as f:
        lines = f.readlines()

        numnode = int(lines[0].split()[0]) # number of nodes
        numelem = int(lines[0].split()[1]) # number of elements
        numfixnodes = int(lines[0].split()[2]) # number of fixed nodes
        ntype = int(lines[0].split()[3]) # if ntype == 1 plane stress; if ntype = 2 plane strain
        numnode_elem = int(lines[0].split()[4]) # number of nodes per element
        numdof = int(lines[0].split()[5]) # degrees of freedom
        numdim = int(lines[0].split()[6]) # dimension of a problem
        numgauss = int(lines[0].split()[7]) # number of integration points
        numstress = int(lines[0].split()[8]) # 
        nummat = int(lines[0].split()[9]) # number of materials 
        numprop = int(lines[0].split()[10]) # number of material properties
        
        elements = np.zeros((numelem,numnode_elem),int)
        for i in range(1,numelem+1):
            elements[i-1,0:numnode_elem+1] = (lines[i].split())[1:numnode_elem+1]
        
        nodes = np.zeros((numnode,numdim),float)
        for i in range(1+numelem,1+numelem+numnode):
            nodes[i-1-numelem,0] = float((lines[i].split())[1])
            nodes[i-1-numelem,1] = float((lines[i].split())[2])
         
        
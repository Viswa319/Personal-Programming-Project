# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import matplotlib.pyplot as plt
import numpy as np
def plot_nodes_and_boundary(nodes):
    """
    Function to plot field nodes and boundary edges

    Parameters
    ----------
    nodes : Array of float64, size(numnode,nx)
        coordinates of all field nodes.

    Returns
    -------
    None.

    """
    fig1,ax1 = plt.subplots()
    # Boundary points
    bot = np.where(nodes[:,1] == min(nodes[:,1]))[0] # bottom edge nodes
    left = np.where(nodes[:,0] == min(nodes[:,0]))[0] # left edge nodes
    right = np.where(nodes[:,0] == max(nodes[:,0]))[0] # right edge nodes
    top = np.where(nodes[:,1] == max(nodes[:,1]))[0] # top edge nodes
    
    # scatters all nodes
    ax1.scatter(nodes[:,0], nodes[:,1], c ='b',s=10)
    # plot top edge nodes
    ax1.plot(nodes[top][:,0],nodes[top][:,1],c ='b')
    # plot bottom edge nodes
    ax1.plot(nodes[bot][:,0],nodes[bot][:,1],c ='b')
    # plot right edge nodes
    ax1.plot(nodes[right][:,0],nodes[right][:,1],c ='b')
    # plot left edge nodes
    ax1.plot(nodes[left][:,0],nodes[left][:,1],c ='b')
    
    ax1.set_title('Nodes')
    ax1.set_xlabel('x[m]')
    ax1.set_ylabel('y[m]')
    plt.show()

def plot_nodes_with_mesh_grid(nodes,nodes_q,ndivxq,ndivyq):
    """
    Function for plotting field nodes witihn background mesh
    
    Parameters
    ----------
    nodes : Array of float64, size(numnode,nx)
        coordinates of all field nodes.

    Returns
    -------
    None.

    """
    Xq, Yq = np.meshgrid(np.linspace(min(nodes_q[:,0]), max(nodes_q[:,0]), ndivxq), np.linspace(min(nodes_q[:,1]), max(nodes_q[:,1]), ndivyq))
    fig2,ax2 = plt.subplots()
    ax2.scatter(nodes[:,0], nodes[:,1],c='b',s=10)
    ax2.plot(Xq,Yq,c='k',ls = '--')
    ax2.plot(np.transpose(Xq), np.transpose(Yq),c='k',ls = '--')
    ax2.set_title('Nodes with mesh grid')
    ax2.set_xlabel('x[m]')
    ax2.set_ylabel('y[m]')

def plot_deformation_with_nodes_boundary(nodes,Disp_anal,Disp_EFG):
    """
    Function to plot deformation of a problem with boundaries

    Parameters
    ----------
    nodes : Array of float64, size(numnode,nx)
        coordinates of all field nodes.
    Disp_anal : Array of float64, size(2,numnode)
        final numerical displacements.
    Disp_EFG : Array of float64, size(2,numnode)
        final analytical displacements.

    Returns
    -------
    None.

    """
    alpha = 100
    numnode = len(nodes)
    X_disp = []
    Y_disp = []
    X_anal = []
    Y_anal = []
    for i in range(0, numnode):
        X_disp.append(nodes[i][0]+alpha*Disp_EFG[0][i])
        Y_disp.append(nodes[i][1]+alpha*Disp_EFG[1][i])
        X_anal.append(nodes[i][0]+alpha*Disp_anal[0][i])
        Y_anal.append(nodes[i][1]+alpha*Disp_anal[1][i])
    # Boundary points
    bot = np.where(nodes[:,1] == min(nodes[:,1]))[0] # bottom edge nodes
    left = np.where(nodes[:,0] == min(nodes[:,0]))[0] # left edge nodes
    right = np.where(nodes[:,0] == max(nodes[:,0]))[0] # right edge nodes
    top = np.where(nodes[:,1] == max(nodes[:,1]))[0] # top edge nodes
    
    fig3,ax3 = plt.subplots()
    # plot left edge nodes
    ax3.plot(nodes[left][:,0],nodes[left][:,1],c ='b')
    # plot top edge nodes
    ax3.plot(nodes[top][:,0],nodes[top][:,1],c ='b')
    # plot bottom edge nodes
    ax3.plot(nodes[bot][:,0],nodes[bot][:,1],c ='b')
    # plot right edge nodes
    ax3.plot(nodes[right][:,0],nodes[right][:,1],c ='b')

    
   # scatter plot for numerically solved deformation nodes
    ax3.scatter(X_disp,Y_disp,c='b',marker = '_',s = 100,label = 'Displacement EFG')
    # scatter plot for analytically solved deformation nodes
    ax3.scatter(X_anal,Y_anal,c='y',marker = '|',s = 100,label = 'Displacement Analytical')
    # scatter plot for all field nodes
    ax3.scatter(nodes[:,0], nodes[:,1],c='b',s = 1,marker = ',', label = 'Nodes without displacement')
   
    plt.legend(loc = 1,bbox_to_anchor =(1.55, 0.65),fancybox=True)
    ax3.set_title('Deformation')
    ax3.set_xlabel('x[m]')
    ax3.set_ylabel('y[m]')
    plt.show()
    

def plot_deflection(nodes,Disp_anal,Disp_EFG):
    """
    Function to plot deflection of beam along y = 0

    Parameters
    ----------
    nodes : Array of float64, size(numnode,nx)
        coordinates of all field nodes.
    Disp_anal : Array of float64, size(2,numnode)
        final analytical displacements.
    Disp_EFG : Array of float64, size(2,numnode)
        final numerical displacements.

    Returns
    -------
    None.

    """
    y_0 = np.where(nodes[:,1] == 0)[0] # bottom edge nodes
    fig1,ax1 = plt.subplots()
    ax1.plot(nodes[y_0][:,0],Disp_EFG[:,y_0][1],ls='--',label = 'EFG')
    ax1.plot(nodes[y_0][:,0],Disp_anal[:,y_0][1],ls='-.',label = 'Analytical')
    ax1.set_title('Deflection at y = 0')
    ax1.set_xlabel('x[m]')
    ax1.set_ylabel('Displacement $u_y$[m]')
    plt.legend(loc = 0)
    plt.show()
    

def plot_stress(nodes,Stress_anal,Stress_EFG,point):
    """
    Function to plot shear stress of beam at given point.
    If any other stress is required, then change in plots
    
    Parameters
    ----------
    nodes : Array of float64, size(numnode,nx)
        coordinates of all field nodes.
    Stress_anal : Array of float64, size(3,numnode)
        Analytical Stress for all field nodes.
    Stress_EFG : Array of float64, size(3,numnode)
        Numerical Stress for all field nodes.

    Returns
    -------
    None.

    """
    y_0 = np.where(nodes[:,0] == point)[0] # bottom edge nodes
    fig1,ax1 = plt.subplots()
    ax1.plot(nodes[y_0][:,1],Stress_EFG[:,y_0][2],ls='--',label = 'EFG')
    ax1.plot(nodes[y_0][:,1],Stress_anal[:,y_0][2],ls='-.',label = 'Analytical')
    ax1.set_title('Normal stress $\sigma$$_{xy}$ at x = L/2')
    ax1.set_xlabel('y[m]')
    ax1.set_ylabel('Normal stress $\sigma$$_{xy}$[$N/m^2$]')
    plt.legend(loc = 0)
    plt.show()
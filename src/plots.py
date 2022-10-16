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
    nodes : Array of float64, size(num_node,num_dim)
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
    ax1.set_xlabel('x[mm]')
    ax1.set_ylabel('y[mm]')


def plot_displacement_contour(nodes,Disp,tot_inc):
    """
    Function to plot contour plot of displacement in vertical direction for tensile and in horizontal direction for shear.

    Parameters
    ----------
    nodes : Array of float64, size(num_node,num_dim)
        Co-ordinates of all field nodes.
    Disp : Array of float64, size(num_node)
            Displacement in vertical direction for tensile and in horizontal direction for shear.
    tot_inc : float64
        Applied displacement.

    Returns
    -------
    None.

    """
    fig1,ax1 = plt.subplots()
    plt.tricontourf(nodes[:,0], nodes[:,1], Disp)
    ax1.set_title('Displacment $u_y$')
    ax1.set_xlabel('x[mm]')
    ax1.set_ylabel('y[mm]')
    plt.colorbar()
    plt.savefig(f'plots\disp={round(tot_inc,5)}_tensile.png'.format(round(tot_inc,5)),dpi=600,transparent = True)

def plot_field_parameter(nodes,phi,tot_inc):
    """
    Function to plot contour plot of field parameter (phi)

    Parameters
    ----------
    nodes : Array of float64, size(num_node,num_dim)
        coordinates of all field nodes.
    phi : Array of float64, size(num_tot_var_phi)
            Phase field order parameter.
    tot_inc : float64
        Applied displacement.

    Returns
    -------
    None.

    """
    fig1,ax1 = plt.subplots()
    plt.tricontourf(nodes[:,0], nodes[:,1], phi)
    ax1.set_title(f'Field parameter $\phi$ at $u$ = {round(tot_inc,5)}'.format(round(tot_inc,5)))
    ax1.set_xlabel('x[mm]')
    ax1.set_ylabel('y[mm]')
    plt.colorbar()
    plt.savefig(f'plots\Phi_u={round(tot_inc,5)}.png'.format(round(tot_inc,5)),dpi=600,transparent = True)

def plot_load_displacement(disp,force):
    """
    Function to generate load vs displacement curve

    Parameters
    ----------
    disp : Array of float64, size(num_step)
        Displacements.
    force : Array of float64, size(num_step)
        Load.

    Returns
    -------
    None.

    """
    fig,ax = plt.subplots()
    plt.plot(disp,force)
    ax.set_title('Load vs displacement')
    ax.set_xlabel('displacement [mm]')
    ax.set_ylabel('load [N]')
    plt.savefig('plots\load_displacement.png',dpi=600,transparent = True)
    
def plot_deformation_with_nodes(nodes,X_Disp,Y_Disp):
    """
    Function to plot deformation of a problem, This plot is used for one element,
    and of course can be used for many elements.

    Parameters
    ----------
    nodes : Array of float64, size(num_node,num_dim)
        coordinates of all field nodes.
    X_Disp : Array of float64, size(num_node)
        displacement in x-direction.
    Y_Disp : Array of float64, size(num_node)
        displacement in y-direction.

    Returns
    -------
    None.

    """
    import numpy as np
    alpha = 1
    numnode = len(nodes)
    X_disp = []
    Y_disp = []
    nodes_disp = np.zeros(np.shape(nodes))
    for i in range(0, numnode):
        X_disp.append(nodes[i][0]+alpha*X_Disp[i])
        Y_disp.append(nodes[i][1]+alpha*Y_Disp[i])
    nodes_disp[:,0] = nodes[:,0]+alpha*X_Disp
    nodes_disp[:,1] = nodes[:,1]+alpha*Y_Disp
    fig3,ax3 = plt.subplots()
    # scatter plot for numerically solved deformation nodes
    ax3.scatter(X_disp,Y_disp,c='r',marker = 'o',s = 20,label = 'Nodes after deformation')
    # scatter plot for all field nodes
    ax3.scatter(nodes[:,0], nodes[:,1],c='b',s = 20,marker = 'o', label = 'Nodes before deformation')
    
    # Boundary points
    bot = np.where(nodes[:,1] == min(nodes[:,1]))[0] # bottom edge nodes
    left = np.where(nodes[:,0] == min(nodes[:,0]))[0] # left edge nodes
    right = np.where(nodes[:,0] == max(nodes[:,0]))[0] # right edge nodes
    top = np.where(nodes[:,1] == max(nodes[:,1]))[0] # top edge nodes

    # plot top edge nodes
    ax3.plot(nodes[top][:,0],nodes[top][:,1],c ='b',linestyle = '--')
    # plot bottom edge nodes
    ax3.plot(nodes[bot][:,0],nodes[bot][:,1],c ='b',linestyle = '--')
    # plot right edge nodes
    ax3.plot(nodes[right][:,0],nodes[right][:,1],c ='b',linestyle = '--')
    # plot left edge nodes
    ax3.plot(nodes[left][:,0],nodes[left][:,1],c ='b',linestyle = '--')
    
    # plot top edge nodes
    ax3.plot(nodes_disp[top][:,0],nodes_disp[top][:,1],c ='r',linestyle = '-.')
    # plot bottom edge nodes
    ax3.plot(nodes_disp[bot][:,0],nodes_disp[bot][:,1],c ='r',linestyle = '-.')
    # plot right edge nodes
    ax3.plot(nodes_disp[right][:,0],nodes_disp[right][:,1],c ='r',linestyle = '-.')
    # plot left edge nodes
    ax3.plot(nodes_disp[left][:,0],nodes_disp[left][:,1],c ='r',linestyle = '-.')
    
    plt.legend(loc='center')
    ax3.set_title('Deformation')
    ax3.set_xlabel('x[mm]')
    ax3.set_ylabel('y[mm]')
    plt.savefig('plots\deform_one_element_1.png',dpi=600,transparent = True)
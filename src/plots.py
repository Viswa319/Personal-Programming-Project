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


def plot_field_parameter(nodes,phi,tot_inc):
    """
    Function to plot contour plot of field parameter (phi)

    Parameters
    ----------
    nodes : Array of float64, size(numnode,num_dim)
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

def plot_load_displacement_mono(disp,force):
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
    plt.savefig('plots_mono\load_displacement.png',dpi=600,transparent = True)

def plot_field_parameter_mono(nodes,phi,tot_inc):
    """
    Function to plot contour plot of field parameter (phi)

    Parameters
    ----------
    nodes : Array of float64, size(numnode,num_dim)
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
    plt.savefig(f'plots_mono\Phi_u={round(tot_inc,5)}.png'.format(round(tot_inc,5)),dpi=600,transparent = True)
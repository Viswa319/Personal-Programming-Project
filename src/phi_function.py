# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
import numpy as np
import matplotlib.pyplot as plt
def phi_function():
    """
    This function generates a plot for phase-field order parameter varying the length parameter
    
    Returns
    -------
    None.

    """
    x = np.linspace(-5,5,101)
    l_0 = [0.001,0.05,0.1,0.5,1.0]
    phi = {}
    for i in l_0:
        Phi = np.zeros(len(x))
        for j in range(0,len(x)):
            Phi[j] = np.exp(-abs(x[j])/i)
        phi[i] = Phi.tolist()
        
    fig,ax = plt.subplots()
    for i in phi:
        plt.plot(x,phi[i],label = f'$l$ = {i}')
        ax.set_title('Variation of phase-field parameter $\phi$')
        ax.set_xlabel('x')
        ax.set_ylabel('$\phi$')
        plt.legend()
        plt.savefig('phi_function.png',dpi=600,transparent = True)

phi_function()
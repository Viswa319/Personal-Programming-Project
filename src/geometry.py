# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
def quadrature_coefficients(numgauss):
    """
    Function to get Gaussian quadrature points and weights  

    Parameters
    ----------
    numgauss : int
        number of Guass points for background integration.

    Returns
    -------
    Points : Array of float64, size(num_dim,num_Gauss**num_dim)
            Gauss points used for integration.
    Weights : Array of float64, size(num_Gauss**num_dim)
            Weights for Gauss points used for integration.

    """
    import numpy.polynomial.legendre as ptwt
    # Gauss quadrature points and weights are computed using inbuilt funtion in numpy 
    PtsWts = ptwt.leggauss(numgauss)
    points = PtsWts[0]
    weights = PtsWts[1]
    Points = []
    Weights = []
    # Computing Gauss points and weights in 2-dimension
    for i in range(0,len(points)):
        for j in range(0,len(points)):
            Points.append([points[i],points[j]])
            Weights.append(weights[i]*weights[j])
    return Points,Weights
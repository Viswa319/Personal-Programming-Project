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
    Points : array of float64
        Gaussian quadrature points.
    Weights : array of float64
        weights for respective points.

    """
    import numpy.polynomial.legendre as ptwt
    PtsWts = ptwt.leggauss(numgauss)
    points = PtsWts[0]
    weights = PtsWts[1]
    Points = []
    Weights = []
    for i in range(0,len(points)):
        for j in range(0,len(points)):
            Points.append([points[i],points[j]])
            Weights.append(weights[i]*weights[j])
    return Points,Weights
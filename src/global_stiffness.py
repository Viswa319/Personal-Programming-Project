# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
def global_stiffness(index,elem_K,global_K):
    for irow in range(0,len(index)):
        irs = int(index[irow])
        for icol in range(0,len(index)):
            ics = int(index[icol])
            global_K[irs,ics] = global_K[irs,ics]+elem_K[irow,icol]
    return global_K
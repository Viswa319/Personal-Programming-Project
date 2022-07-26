# *************************************************************************
#                 Implementation of phase-field model of ductile fracture
#                      Upadhyayula Sai Viswanadha Sastry
#                         Personal Programming Project
#                               65130
# *************************************************************************
class global_stiffness_assembly:
    def global_stiffness(self,index,elem_K,global_K):
        '''
        

        Parameters
        ----------
        index : Array of float64, size(num_dof_elem)
            a vector which consists indices of global matrix for respective element.
        elem_K : TYPE
            DESCRIPTION.
        global_K : TYPE
            DESCRIPTION.

        Returns
        -------
        global_K : TYPE
            DESCRIPTION.

        '''
        for i in range(0,len(index)):
            k = int(index[i])
            for j in range(0,len(index)):
                l = int(index[j])
                global_K[k,l] = global_K[k,l]+elem_K[i,j]
            
        return global_K
    def global_stiffness_coupled(self,index_1,index_2,elem_K,global_K):
        '''
        

        Parameters
        ----------
        index_1 : Array of float64, size(num_dof_elem)
            a vector which consists indices of global matrix for respective element related to first fireld parameter (u).
        index_2 : Array of float64, size(num_dof_elem)
            a vector which consists indices of global matrix for respective element related to first fireld parameter (phi).
        elem_K : TYPE
            DESCRIPTION.
        global_K : TYPE
            DESCRIPTION.

        Returns
        -------
        global_K : TYPE
            DESCRIPTION.

        '''
        for i in range(0,len(index_1)):
            k = int(index_1[i])
            for j in range(0,len(index_2)):
                l = int(index_2[j])
                global_K[k,l] = global_K[k,l]+elem_K[i,j]
        return global_K
    
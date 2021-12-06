# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 11:37:17 2021

@author: rhydi
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 12:17:18 2021

@author: rhydi
"""
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from cython.parallel cimport prange
cimport openmp
import cython
from body_setup import n_body_coords

np.import_array()
DTYPE = float
ctypedef np.float_t DTYPE_t
@cython.boundscheck(False)

@cython.cclass
class gravity_sim_calc:
    
    # Cythonised
    def calculate_single_force_vector(self, np.ndarray body_i_array, np.ndarray body_j_array, double epsilon):
        """
        Takes the dictionaries of 2 bodies, labelled i and j and calculates
        the force vector between them in the i and j basis.
    
        Parameters
        ----------
        body_dict_i : dict
            Dictionary for body 1 in this interaction.
        body_dict_j : dict
            Dictionary for body 2 in this interaction.
    
        Returns
        -------
        F_ij : Vector
            Force vecter between body i and j.
    
        """        
        cdef double one_over_G = 1.498333853*10**(10) # m^-3 kg^1 s^2 Graviational constant         
        r_i_vector, M_i = body_i_array[0], body_i_array[-1]
        r_j_vector, M_j = body_j_array[0], body_j_array[-1]
        displacement_vector_ij = r_i_vector-r_j_vector
        square_of_ij_distance = displacement_vector_ij.dot(displacement_vector_ij)
        unit_displacement_vector = displacement_vector_ij/sqrt(square_of_ij_distance)
        F_ij_magnitude = (-1*M_i*M_j)/(one_over_G*(square_of_ij_distance + epsilon))
        F_ij_vector = F_ij_magnitude*unit_displacement_vector
        return F_ij_vector
    
    # Not yet cythonised
    def force_matrix(self, np.ndarray array_bodies_details, double epsilon):
        """
        Takes the dictionaries of each body to fill out the force matrix.
        Matrix elements are the force vectors as calculated in
        calculate_single_force_vector:
            
            Vector F_ij, i across, j down is logged line by line. 
            
            Since F_ii = 0 and F_ji = -F_ij it uses the following setup:
                i<j:
                    Appends -F_ij for F_ji'th element
                i>j:
                    Calculates and appends F_ij
                i==j:
                    Appends [0, 0]
    
        Parameters
        ----------
        array_bodies_coords : array
            Contains the position vectors for each body.
    
        Returns
        -------
        force_matrix : array
            Matrix containing each of the force elements.
    
        """                
        cdef int threads = self.threads
        cdef int no_bodies = len(array_bodies_details)
        cdef int dimensions = self.dimensions
        cdef int i, j
        cdef int M_i, M_j
        cdef double initial, final
        cdef double square_of_ij_distance
        cdef double F_ij_magnitude
        cdef np.ndarray force_matrix = np.empty((no_bodies, no_bodies), dtype=np.object)        
        
        initial = openmp.omp_get_wtime()        
        for j in prange(no_bodies, nogil=True, num_threads=threads):
            for i in range(j+1, no_bodies):       
                force_matrix[j,i] = 1
                
        # final = openmp.omp_get_wtime()
        
        # for j in range(no_bodies):
        #     for i in range(no_bodies):
                
        #         if i<j:
        #             force_matrix[j, i] = -force_matrix[i, j]
                
        #         elif i==j:
        #             force_matrix[j, i] = np.zeros(dimensions)
                
        #         elif i>j:
        #             pass   
        
        self.force_matrix = force_matrix
    
    # Cythonised
    def time_evolve_bodies_one_step(self):
        """
        Takes the force matrix and adds it up to find the net force on each body.
        Applies this to the previously recorded positions and velocities via:
            V_new = V_old + F*dt/M
            S_new = S_old + V_old*dt + F*dt**2/2*M
        Does so for each component creating a new vector. This new vector
        replaces the current coordinates in the array_of_body_dictionaries
        and appends the log dictionary, to save memory. 
        
        Parameters
        ----------
        force_matrix_ev : Array
            Force matrix as calculated in force_matrix.
        array_bodies_dicts_ev : Array
            Contains dictionaries for each body.
        dt : float
            Time step.
    
        Returns
        -------
        None.
    
        """
        cdef:
            object force_matrix_ev = self.force_matrix
            object array_bodies_ev = self.array_bodies_details
            object position_vectors_bodies = array_bodies_ev[:, 0]
            object velocity_vectors_bodies = array_bodies_ev[:, 1]
            object reciprocal_mass_bodies = 1/array_bodies_ev[:, 2]
            object net_force_vector_bodies = force_matrix_ev.sum(axis=0)
            object velocity_vectors_bodies_updated, position_vectors_bodies_updated
            double dt = self.dt
            
        
        velocity_vectors_bodies_updated = velocity_vectors_bodies + net_force_vector_bodies*dt*reciprocal_mass_bodies
        position_vectors_bodies_updated = position_vectors_bodies + velocity_vectors_bodies*dt + net_force_vector_bodies*(dt**2)*reciprocal_mass_bodies/2
        
        return position_vectors_bodies_updated, velocity_vectors_bodies_updated
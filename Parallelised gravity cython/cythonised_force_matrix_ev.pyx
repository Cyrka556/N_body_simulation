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

np.import_array()
DTYPE = float
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)
def get_bodies(int number_of_bodies, int dimensions):
    cdef double space_boundary = 10**12
    cdef double vel_boundary = 0
    cdef double min_body_mass = 6 # x10^24
    cdef double max_body_mass = 32 # x10^24
    cdef double[:] arr_body_mass = (max_body_mass-min_body_mass)*np.random.random((number_of_bodies))-min_body_mass
    cdef double[:,:] arr_body_pos = (2*space_boundary)*np.random.random((number_of_bodies,dimensions))-space_boundary
    cdef double[:,:] arr_body_vel = (2*vel_boundary)*np.random.random((number_of_bodies,dimensions))-vel_boundary
    cdef np.ndarray np_arr_body_pos = np.array(arr_body_pos, dtype=np.double)
    cdef np.ndarray np_arr_body_vel = np.array(arr_body_vel, dtype=np.double)
    cdef np.ndarray np_arr_body_mass = np.array(arr_body_mass, dtype=np.double)
    
    return arr_body_pos, arr_body_vel, arr_body_mass
    
# Cythonised
@cython.boundscheck(False)
@cython.wraparound(False)
def force_matrix(double[:,:] body_array_positions, double[:] body_array_mass, double epsilon, int dimensions, int THREADS):
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
    cdef int no_bodies = body_array_mass.shape[0]
    cdef double[:,:,:] F_matrix_for_filling = np.zeros((no_bodies, no_bodies, dimensions)) 
    cdef double one_over_G = 1.498333853*10**(10) # m^-3 kg^1 s^2 Graviational constant
    cdef double mass_multplier_factor = 1*10**(48)
    cdef int i, j, direction_index, direction_squared_sum, k, l
    cdef double[:] zero_force_vector = np.zeros(dimensions)
    cdef double epsilon_squared = epsilon**2
    cdef double r_ij_squared, r_ij_mag, F_grav_mod_coef, F_grav_ij_component
    
    initial = openmp.omp_get_wtime()
    for j in range(no_bodies):
        for i in prange(j+1, no_bodies, nogil=True, num_threads=THREADS):
            r_ij_squared = 0
            r_ij_mag = 0
            for direction_squared_sum in range(dimensions):
                r_ij_squared += (body_array_positions[i][direction_squared_sum]-body_array_positions[j][direction_squared_sum])**2
            r_ij_mag += sqrt(r_ij_squared)
            
            F_grav_mod_coef = 0
            F_grav_mod_coef += (-1*mass_multplier_factor*body_array_mass[i]*body_array_mass[j])/(one_over_G*r_ij_mag*(r_ij_squared + epsilon_squared))
            
            for direction_index in range(dimensions):
                F_grav_ij_component = 0
                F_grav_ij_component += F_grav_mod_coef*(body_array_positions[i][direction_index]-body_array_positions[j][direction_index])
                F_matrix_for_filling[j][i][direction_index] += F_grav_ij_component
                F_matrix_for_filling[i][j][direction_index] -= F_grav_ij_component
                
    final = openmp.omp_get_wtime()
    print("Elapsed time: {:8.6f} s".format(final-initial))

    np_F_matrix = np.array(F_matrix_for_filling, dtype=np.double)
    
    return np_F_matrix

# Cythonised - needs redo for 3d zeros matrix
def time_evolve_bodies_one_step(double[:,:] current_bodies_pos, double[:,:] current_bodies_vels, double[:] arr_body_masses, double epsilon, double dt, int dimensions, int THREADS):
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
    
    cdef np.ndarray np_current_bodies_pos = np.array(current_bodies_pos)
    cdef np.ndarray np_current_bodies_vels = np.array(current_bodies_vels)
    cdef np.ndarray np_arr_body_masses = np.array(arr_body_masses)
    cdef np.ndarray F_grav_ij_matrix = force_matrix(current_bodies_pos, arr_body_masses, epsilon, dimensions, THREADS)
    cdef np.ndarray net_force_vector_bodies = np.sum(F_grav_ij_matrix, axis=1)
    cdef double mass_multplier_factor = 1*10**(24)
    
    np_current_bodies_vels += ((net_force_vector_bodies*dt).T/(mass_multplier_factor*np_arr_body_masses)).T
    np_current_bodies_pos += np_current_bodies_vels*dt + ((net_force_vector_bodies*(dt**2)).T/(2*mass_multplier_factor*np_arr_body_masses)).T
    
    cdef double[:,:] new_bodies_vels = np_current_bodies_vels
    cdef double[:,:] new_bodies_pos = np_current_bodies_pos
    
    return new_bodies_vels, new_bodies_pos
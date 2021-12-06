# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 09:06:24 2021

@author: rhydi
"""
import time
import math 
from random import randint
import numpy as np 

class numerical_tools():
    """Class containing a list of useful tools for the simulation.
    """
    
    def negative_of_vector(self, vector):
        """
        Returns the vector multipled by -1.

        Parameters
        ----------
        vector : Array
            Array containing N vector components.

        Returns
        -------
        Array
            Flipped vector.

        """
        return [-1*vector_component for vector_component in vector]
    
    def distance_details_between_vectors(self, vector_1, vector_2):
        """
        Takes two vectors of N dimensions.
        Processes them in the following steps:
            Checks they are of equal length, continues calculation if they are 
            and raises an error if not.
            
            For each component, subtracts vector_1 by vector_2 and appends to
            an array, creating a displacement array. Also sums the squares of
            each component for a distance scalar
            
            Divides each component in the displacement vector by the distance
            scalar to create a unit vector.

        Parameters
        ----------
        vector_1 : Array
            Vector for body 1.
        vector_2 : Array
            Vector for body 2.

        Raises
        ------
        ValueError
            If vectors are of equal length, returns this problem to the user.

        Returns
        -------
        unit_vector : Array
            Unit displacement vector for the two bodies.
        distance_sum_of_squares : float
            Square of the distance.
        distance_scalar : float
            The distance.

        """
        if len(vector_1) == len(vector_2):
            displacement_vector = []
            distance_sum_of_squares = 0
            for component in range(len(vector_1)):
                q_i_difference = vector_1[component] - vector_2[component]
                displacement_vector.append(q_i_difference)
                distance_sum_of_squares += q_i_difference**2
            distance_scalar = math.sqrt(distance_sum_of_squares)
            unit_vector = [displacement_vector_component/distance_scalar for displacement_vector_component in displacement_vector]
            return unit_vector, distance_sum_of_squares, distance_scalar
            
        else:
            raise ValueError('In excecuting vector_distance the two given vectors were of different size')
            
    def vector_magnitude(self, vector):
        """
        Takes vector and calculates the sum of the squares of the components. 

        Parameters
        ----------
        vector : Array
            Vector we want the magnitude sqaured of.

        Returns
        -------
        sum_of_squares : float
            Sum of squares of vector components.

        """
        sum_of_squares = 0
        for vector_component in vector:
            sum_of_squares += vector_component**2
        return math.sqrt(sum_of_squares)
    
    def net_force_from_matrix(self, force_matrix_for_sum, dimensions_of_vectors):
        """
        Takes the force matrix and calculates the net force vector on each body.
        In the force matrix, each row contains Fij for constant j, the force 
        body i experiences from body j. To sum the net force acting on a single 
        body, Fij for constant i is needed. Exploiting the fact that the matrix
        is diagonally symmetric summing the columns is equivilent to subtracting
        the rows, which is done here. 

        Parameters
        ----------
        force_matrix_for_sum : Array
            Array containing arrays which contain the force vectors (also arrays).
        dimensions_of_vectors : int
            Number of dimensions being considered.

        Returns
        -------
        net_forces_on_each_body : Array
            Array containing the net force vectors.

        """
        net_forces_on_each_body = []
        for force_vector_array in force_matrix_for_sum:
            array_component_sums = [0 for _ in range(dimensions_of_vectors)]
            for force_vector in force_vector_array:
                for direction_considered in range(len(array_component_sums)):
                    array_component_sums[direction_considered] -= force_vector[direction_considered]
            net_forces_on_each_body.append(array_component_sums)
        return net_forces_on_each_body
    
class n_body_coords():
    """Creates bodies for the simulation.
    Calculate force for a timestep"""
      
    def randomly_generate_one_body_details(self):
        """
        Creates a dictionary containing: the initial coordinates, Nd Cartesian;
        initial velocity, in x and y direction; and a gravitational mass. These
        are randomly generated from a range.
        Distance:
            Spans (approximate) diameter of milk-way with 0 at the centre.
        Velocity:
            Currently set to 0
        Mass:
            Taken to be similar to planets, from approx 1 to 32 earth masses.
            
        Returns
        -------
        Dict
            Contains randomly generated information about the body.

        """
        initial_position = np.array([float(randint(int(-4*10**12), int(4*10**12))) for i in range(self.dimensions)], dtype=object) # x, y (m)
        initial_velocity = np.array([0, 0], dtype=object) # x, y dirs (m/s)
        mass = randint(6*10**24, 32*6*10**24) # kg
        
        array_body = np.array([initial_position, initial_velocity,  mass], dtype=object)
        return array_body, initial_position

    def generate_n_random_bodies(self, number_of_bodies, dimensions):
        """
        Creates an array containing number_of_bodies dctionaries as created in
        randomly_generate_one_body_details.

        Parameters
        ----------
        number_of_bodies : int
            Number of bodies to be simulated.
        dimensions : int
            Number of dimensions to consider.

        Returns
        -------
        array_of_body_dicts : Array
            Contains dictionaries for each randomly generated body.

        """
        self.number_of_bodies = number_of_bodies
        self.dimensions = dimensions
        initial_array_of_body_dicts_for_initiation = []
        sum_body_count_vectors = [0 for i in range(dimensions)]
        for body_count in range(number_of_bodies):
            body_dict, initial_position_vector = self.randomly_generate_one_body_details()
            initial_array_of_body_dicts_for_initiation.append(body_dict)
            sum_body_count_vectors = [sum_body_count_vectors[direction_index] + initial_position_vector[direction_index] 
                                      for direction_index in range(len(sum_body_count_vectors))]
        initial_array_of_body_dicts_for_initiation = np.array(initial_array_of_body_dicts_for_initiation)
        self.epsilon = numerical_tools().vector_magnitude([sum_body_count_vectors_components/number_of_bodies for sum_body_count_vectors_components in sum_body_count_vectors]) # Mean initial positions
        self.initial_dictionaries_all_bodies = initial_array_of_body_dicts_for_initiation
        return initial_array_of_body_dicts_for_initiation, self.epsilon
    
    def calculate_single_force_vector(self, body_dict_i, body_dict_j):
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
        G = 6.67408*10**(-11) # m^3 kg^-1 s^-2 Graviational constant
        r_i, M_i = body_dict_i['Coords'][-1], body_dict_i['Mass']
        r_j, M_j = body_dict_j['Coords'][-1], body_dict_j['Mass']
        unit_vector_ij, square_of_ij_distance, ij_distance = numerical_tools().distance_details_between_vectors(r_i, r_j)
        F_ij_magnitude = (-G*M_i*M_j)/(square_of_ij_distance + self.epsilon)
        F_ij_vector = [F_ij_magnitude*unit_vector_component for unit_vector_component in unit_vector_ij]
        return F_ij_vector
    
    def force_matrix(self, array_bodies_dicts):
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
        force_matrix = []
        for j in range(self.number_of_bodies):
            force_row_j = []
            for i in range(self.number_of_bodies):
                if i<j:
                    force_row_j.append(numerical_tools().negative_of_vector(force_matrix[i][j]))
                elif i>j:
                    F_ij_vector = self.calculate_single_force_vector(array_bodies_dicts[i], array_bodies_dicts[j])
                    force_row_j.append(F_ij_vector)
                elif i == j:
                    force_row_j.append([0,0])
            force_matrix.append(force_row_j)
                    
        return force_matrix
    
    def time_evolve_bodies_one_step(self, force_matrix_ev, array_bodies_dicts_ev, dt):
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
        net_force_vector_array = numerical_tools().net_force_from_matrix(force_matrix_ev, self.dimensions)
        for body_index in range(len(array_bodies_dicts_ev)):
            v_new = []
            s_new = []
            relevant_body_dictionary = array_bodies_dicts_ev[body_index]
            body_mass = relevant_body_dictionary['Mass']
            for direction_index in range(self.dimensions):
                v_old_component = relevant_body_dictionary['Velocity'][-1][direction_index]
                s_old_component = relevant_body_dictionary['Coords'][-1][direction_index]
                v_new_component = v_old_component + (net_force_vector_array[body_index][direction_index] * dt)/body_mass
                s_new_component = s_old_component + v_old_component * dt + (net_force_vector_array[body_index][direction_index] * (dt**2))/(2 * body_mass)
                v_new.append(v_new_component)
                s_new.append(s_new_component)
            relevant_body_dictionary['Velocity'].append(v_new)
            relevant_body_dictionary['Coords'].append(s_new)
            
        return net_force_vector_array
        
    @property
    def _N_bodies(self):
        """
        Get the number of bodies

        Returns
        -------
        int
            Number of bodies.

        """
        return self.number_of_bodies
    
    @property
    def _epsilon(self):
        """
        Get epsilon, the dampening factor

        Returns
        -------
        str
            Dampening factor, mean position of all bodies.

        """
        return self.epsilon
    
    @property
    def _dimensions(self):
        """
        Get the number of dimensions

        Returns
        -------
        int
            Number of dimensions.

        """
        return self.dimensions


# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 20:44:47 2021

@author: rhydi
"""
import parallelised_grav_sim as grav_sim
import sys
import numpy as np

def main(number_bodies, dimensions, delta_t, tot_time, THREADS):
    arr_pos, arr_vel, arr_masses = grav_sim.get_bodies(number_bodies, dimensions)
    
    t_var = 0
    while t_var<tot_time:        
        arr_pos, arr_vel = grav_sim.time_evolve_bodies_one_step(arr_pos, arr_vel, arr_masses, 0, delta_t, dimensions, THREADS)
        print(t_var)
        t_var += delta_t

main(1000, 2, 2, 20, 1)

# if int(len(sys.argv)) == 6:
#     no_bodies = sys.argv[1]
#     no_dimensions = sys.argv[2]
#     time_step = sys.argv[3]
#     total_time = sys.argv[4]
#     no_threads = sys.argv[5]
    
#     main(no_bodies, no_dimensions, time_step, total_time, no_threads)
    
# else:
#     print("Usage: {} <No. Bodies> <No. Dimensions> <Time Step> <No. Threads>".format(sys.argv[0]))
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 20:36:26 2021

@author: rhydi
"""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "parallelised_grav_sim",
        ["cythonised_force_matrix_ev.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(name="parallelised_grav_sim",
      ext_modules=cythonize(ext_modules))


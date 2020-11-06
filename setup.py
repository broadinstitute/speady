from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Need to export CFLAGS="-I /usr/local/lib/python3.7/site-packages/numpy/core/include/ $CFLAGS"
setup(ext_modules=cythonize("pearson.pyx", include_path=[numpy.get_include()]))

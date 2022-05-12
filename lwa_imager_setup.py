# Before running, apt-get install cython
# to compile the cintf_tools.so module:
# python cintf_tools_setup.py build_ext --inplace
# if you get the error:
# "ValueError: numpy.ufunc has the wrong size, try recompiling"
# this is an indication that cintf_tools was not properly compiled for 
# your architecture
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
 
import numpy
 
ext = Extension("lwa_imager", ["lwa_imager.pyx"], 
	include_dirs = [numpy.get_include()],
	extra_compile_args=['-fopenmp'],
	extra_link_args=['-fopenmp'],
	)
 
setup(ext_modules=[ext], cmdclass = {'build_ext': build_ext})

from distutils.core import setup, Extension
import numpy.distutils.misc_util

setup(
   ext_modules = [Extension("_c_ints",["_c_ints.c","c_ints/shellpair_quantities.c",
                  "c_ints/one_electron_fundamentals.c", "c_ints/one_electron_vrr.c",
                  "c_ints/one_electron_hrr.c", "c_ints/one_electron_contract.c",
                  "c_ints/one_electron_kinetic.c", "c_ints/two_electron_bound.c",
                  "c_ints/two_electron_fundamentals.c","c_ints/two_electron_vrr.c",
                  "c_ints/two_electron_contract.c","c_ints/two_electron_hrr.c",
                  "c_ints/two_electron_scattering.c","c_ints/spherical_bessel_j.c",
                  "c_ints/angmom_index.c","c_ints/interpolation_table.c"])],
   include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs(),
)


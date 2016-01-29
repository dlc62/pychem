from scipy.special import gamma
from math import sqrt

def normalize(float exponent, ls):
    cdef float lx, ly, lz, expo
    [lx,ly,lz] = ls
    expo = ((lx+ly+lz+1.5)/2.0) / sqrt(gamma(lx+0.5)*gamma(ly+0.5)*gamma(lz+0.5))
    return((2.0*exponent) ** expo)

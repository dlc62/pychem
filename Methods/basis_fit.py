# System libraries
import sys
import numpy
import copy
import math
# Custom-written data modules
from Data import basis
from Data import constants as c

#----------------------------------------------------------#
#  Utility functions
#----------------------------------------------------------#
def overlap(e1,e2,l):
    s = math.pow(2.0*e1,0.75) * math.pow(2.0*e2,0.75) * math.pow(e1+e2,-(1.5+l))
    return s

def cgtf_overlap(cgtfs1,cgtfs2,l):
    overlap_matrix = []
    for cgtf1 in cgtfs1:
       overlap_vector = []
       for cgtf2 in cgtfs2:
          overlap_element = 0.0
          for [e1,c1] in cgtf1:
             for [e2,c2] in cgtf2:
                overlap_element += c1*c2*overlap(e1,e2,l)
          overlap_vector.append(overlap_element)
       overlap_matrix.append(overlap_vector)
    return overlap_matrix

#----------------------------------------------------------#
#  Main routine
#----------------------------------------------------------#
def do(molecule, MOs, new_basis_name):

    # Construct fitting vectors to express each CGTF in the old basis
    # as a linear combination of CGTFs in the new basis

    fit_coeffs_by_atom = []; new_basis = []
    for atom in molecule.Atoms:
       new_cgtfs = basis.get[new_basis_name][atom.Label]
       new_basis.append(new_cgtfs)
       fit_coeffs = []
       for cgtf in atom.Basis:
          ref_l = cgtf.AngularMomentum
          ref_primitives = cgtf.Primitives
          fitting_functions = []
          for new_cgtf in new_cgtfs:
             new_l = new_cgtf[0]
             new_primitives = new_cgtf[1:] 
             if new_l == ref_l:
                fitting_functions.append(new_primitives)
          if fitting_functions != []:
             S_matrix = numpy.array(cgtf_overlap(fitting_functions,fitting_functions,ref_l))
             T_vector = numpy.array(cgtf_overlap([ref_primitives],fitting_functions,ref_l)[0]) 
             coeff_vector = numpy.linalg.solve(S_matrix,T_vector).flatten().tolist()
             fit_coeffs.append(coeff_vector) 
          else:
             fit_coeffs.append([])
       fit_coeffs_by_atom.append(fit_coeffs)
           
    # Compute where to place new coefficients according to angular momentum of functions in new basis

    l_sort_by_atom = []
    position = 0; 
    for functions in new_basis:
       l_sort = [[] for i in range(0,10)]
       for function in functions:
          new_l = function[0]
          l_sort[new_l].append(position)
          if new_l == 2 and molecule.CartesianD == True:
             n_new_functions = c.nAngMomCart[new_l]
          else:
             n_new_functions = c.nAngMomSpher[new_l] 
          position += n_new_functions
       l_sort_by_atom.append(l_sort)
    n_new_MOs = position

    # Expand out MOs from old basis into new 

    new_MOs = numpy.zeros((n_new_MOs,n_new_MOs))

    for (i,atom) in enumerate(molecule.Atoms):
       for (j,cgtf) in enumerate(atom.Basis):
          ref_l = cgtf.AngularMomentum
          n_l = cgtf.NAngMom
          c_fit = fit_coeffs_by_atom[i][j]
          l_sort = l_sort_by_atom[i][ref_l]
          # spread out new coefficients for new functions
          for (cf,index) in zip(c_fit,l_sort):
             cMO = MOs[index:index+n_l,:]
             for n in xrange(len(cMO)):
                for m in xrange(len(cMO[n])):
                   new_MOs[index+n,m] += cf*cMO[n,m]

    return new_MOs

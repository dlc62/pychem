# System libraries
from copy import deepcopy
import pickle
import numpy
from numpy import dot

# --------------------- Helpful subroutines ---------------------------# 

# Used in pychem.py

def excite(matrix, occupancy, NElectrons):
# This function permutes the molecular orbitals for excited states 
# so that the newly occupied orbitals come before newly unoccupied ones
    new_matrix = deepcopy(matrix)
    frm = []                        # list contains indexes of orbitals to be excited from
    to = []                         # list contains indexes of orbitals to be excited to
    for i in range(NElectrons):
        if occupancy[i] == 0:
            frm.append(i)
    for i in range(NElectrons,len(occupancy)):
        if occupancy[i] == 1:
            to.append(i)
    for i in range(len(to)):
        new_matrix[:,[frm[i],to[i]]] = new_matrix[:,[to[i],frm[i]]]
    return new_matrix

# Used everywhere 

def store(data_type_suffix, data, section_name, basis_set=None):
    if basis_set == None:
       # will only store data for current basis set, overwriting previous data
       section_name = section_name + '.'
    else:
       # store data for all basis sets
       section_name = section_name + '_'
       basis_set = basis_set + '.'
    pickle.dump(data, open(section_name + basis_set + data_type_suffix,'wb')) 

def fetch(data_type_suffix, section_name, basis_set=None):
    if basis_set == None:
       section_name = section_name + '.'
    else:
       section_name = section_name + '_'
    data = pickle.load(open(section_name + basis_set + data_type_suffix,'rb')) 
    return data

# Used in inputs_structures.py

def remove_punctuation(basis_set):
    basis_set = basis_set.replace('*','s').replace('-','').replace('(','').replace(')','').replace(',','').upper()
    return basis_set

def single_excitations(n_electrons, n_orbitals):
    """Takes the number of electrons of a particular spin and the number
    of orbitals and returns the list of pairs corresponding to all single
    excitations"""
    excitations = []
    n_virtual_orbitals = n_orbitals - n_electrons
    for i in range(1,n_electrons+1):
        for j in range(1,n_virtual_orbitals+1):
            excitations.append([-i,j])
    return excitations

def make_length_equal(list1, list2, place_holder = []):
    diff = len(list1) - len(list2)
    extra = [place_holder for i in range(abs(diff))]
    if diff > 0:
        list2 += extra
    elif diff < 0:
        list1 += extra

# Used in diis.py

def inner_product(mat1, mat2):
    product = mat1.dot(mat2.T)
    return numpy.trace(product)

def eigenvalue_condition_number(matrix):
    eigvals, eigvecs = numpy.linalg.eig(matrix)
    abs_eigvals = [abs(element) for element in eigvals]
    condition_number = max(abs_eigvals) / min(abs_eigvals) 
    return condition_number


# System libraries
from copy import deepcopy
import pickle
import numpy
import scipy
import os

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

def store(data_type_suffix, data, section_name, basis_set=None):
    # Check if the storage dirrectry exists and make it if it doesn't
    if not os.path.isdir("./Store"):
        os.makedirs("./Store")
    if basis_set == None:
       # will only store data for current basis set, overwriting previous data
       section_name = section_name + '.'
    else:
       # store data for all basis sets
       section_name = section_name + '_'
       basis_set = basis_set + '.'
    pickle.dump(data, open("./Store/" + section_name + basis_set + data_type_suffix,'wb'))

def fetch(data_type_suffix, section_name, basis_set=None):
    if basis_set == None:
       section_name = section_name + '.'
    else:
       section_name = section_name + '_'
    data = pickle.load(open( "./Store/" + section_name + basis_set + "." + data_type_suffix,'rb'))
    return data

# Used in inputs_structures.py

def remove_punctuation(basis_set):
    basis_set = basis_set.replace('*','s').replace('-','').replace('(','').replace(')','').replace(',','').upper()
    return basis_set

def single_excitations(ground_occ):
    excitations = []
    occ_stop = sum(ground_occ)        # last occupied orbital
    for i in range(0, occ_stop):
        for j in range(occ_stop, len(ground_occ)):
            excitations.append([i,j])
    return excitations

def double_paired_excitations(ground):
    excitations = []
    for (i, orb1) in enumerate(zip(ground.AlphaOccupancy, ground.BetaOccupancy)):
        if orb1 == (1,1):
            for (j, orb2) in enumerate(zip(ground.AlphaOccupancy, ground.BetaOccupancy)):
                if orb2 == (0,0):
                    excitations.append([i,j])
    return excitations, excitations

def double_excitations(ground):
    alpha_excitations = []; beta_excitations = []
    alpha_singles = single_excitations(ground.AlphaOccupancy)
    beta_singles = single_excitations(ground.BetaOccupancy)
    for excite1 in alpha_singles:
        for excite2 in beta_singles:
            alpha_excitations.append(excite1)      # Will this give mutiple identical
            beta_excitations.append(excite2)       # excitations in some cases?
    return alpha_excitations, beta_excitations

def make_length_equal(list1, list2, placeholder = []):
    diff = len(list1) - len(list2)
    extra = [placeholder for i in range(abs(diff))]
    if diff > 0:
        list2 += extra
    elif diff < 0:
        list1 += extra

# Used in diis.py

def inner_product(mat1, mat2):    # Also used in NOCI
    product = mat1.dot(mat2.T)
    return numpy.trace(product)

def eigenvalue_condition_number(matrix):
    eigvals, eigvecs = numpy.linalg.eig(matrix)
    abs_eigvals = [abs(element) for element in eigvals]
    condition_number = max(abs_eigvals) / min(abs_eigvals)
    return condition_number

# used in NOCI

def resize_array(src, dest, fill=0):
    """ Makes array src the same size as array dest, the old array is embeded in
     the upper right corner and the other elements are zero. Only works for
     for projecting a vector into a vector or a matrix into a matrix """
    old_shape = numpy.shape(src)
    new_array = numpy.full_like(dest, fill)
    if len(old_shape) == 2:              # Matrix
        height, width = old_shape
        new_array[:height, :width] = src
    else:                                # Vector
        length = old_shape[0]
        new_array[:length] = src
    return new_array

def occupied(matrices):
    """ Returns a matrix corresponding to just the occupied MOs"""
    n = sum(matrices.Occupancy)
    size = len(matrices.Occupancy)
    return matrices.MOs[:,0:n]

def distance(density1, density2, molecule):
    """ Calculatest he distance between two sets of MOs using the metric
        from Phys. Rev. Lett. 101 193001 """
    co_density = molecule.Overlap.dot(density2).dot(molecule.Overlap)            # Calculating the covarient density matrix
    den_prod, _ = scipy.linalg.sqrtm(density1.dot(co_density), disp=False)       # Note the sqrtm which isn't in the paper
    return molecule.NElectrons - numpy.real(numpy.trace(den_prod))               # Imaginary part is a machine precision error from sqrtm

def distance_matrix(molecule):
    """ Calculates the distances between all the calculates states using
        the above distance metric """
    distances = [[distance(state1.Total.Density, state2.Total.Density, molecule)
                    for state1 in molecule.States] for state2 in molecule.States]
    return numpy.array(distances)

def randU(n):
    """ Generates a random n x n unitary matrix """
    X = numpy.random.randn(n,n)
    q, r = numpy.linalg.qr(X)
    R = numpy.diag(numpy.sign(numpy.diag(r)))
    return q.dot(R)

def sort_MOs(state, molecule, only_occupied=True):
    """ Sort ths MOs, if only_occupied is True sort just the occupied ones """
    N  = sum(state.Occupancy) if only_occupied else molecule.NOrbitals
    MOs_to_sort = state.MOs[:,:N]
    energies_to_sort = state.Energies[:N]
    indices = energies_to_sort[:N].argsort()
    state.MOs[:,:N] = MOs_to_sort[:,indices]
    state.Energies[:N] = energies_to_sort[indices]

#def visualize_MOs(MOs, basis_set, molecule):
def visualize_MOs(MOs, molecule, basis_set=None):
    from Data.basis import get
    ang_labels = {0: ["s"], 1: ["px", "py", "pz"]}
    if basis_set is None:
        basis_set = molecule.Basis

    # collect the anular momenytum labels of the various cgtos
    atoms = [atom.Label for atom in molecule.Atoms]
    funcs = []; orb_atoms = []
    for atom in atoms:
        cgtos = get[basis_set][atom]
        if len(atom) == 2:
            atom = atom[0] + atom[1].lower()
        for cgto in cgtos:
            funcs += ang_labels[cgto[0]]
            orb_atoms += [atom] * (cgto[0] * 2 + 1)

    # Ensure that the size of the basis matches the size of the MOs
    assert len(funcs) == len(MOs), "Incorect Basis Set For MOs"

    print('')
    for i, orb in enumerate(funcs):
        print("{:>4} {:>3}: {}".format(orb_atoms[i], orb, MOs[i,:]))
    print('')

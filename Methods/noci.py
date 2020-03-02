# System libraries
import numpy as np
from scipy.linalg import eigh as gen_eig
from collections import namedtuple 
from copy import copy
# Custom code
from Util import printf, structures
from Util.structures import Spin 
from Data import constants as const
import hartree_fock as hf
import mp2

#--------------------------------------------------------------#
#                Set up required structures                    # 
#--------------------------------------------------------------#

ZeroOverlap = namedtuple("Zero", "index, spin")

class CoDensityState:
   # # A state-like object used for calculating and storing the pseudo Fock matrix
    def __init__(self, n_orbitals, alpha_density, beta_density):
        self.Alpha = Matrices(n_orbitals)
        self.Beta = Matrices(n_orbitals)
        self.Total = Matrices(n_orbitals,total=True) 
        self.Alpha.Density = alpha_density
        self.Beta.Density = beta_density
        self.Total.Density = alpha_density + beta_density

class Matrices:
    def __init__(self,n_orbitals,total=False):
        self.Density = np.zeros((n_orbitals,) * 2)
        if not total:
            self.Exchange = np.zeros((n_orbitals,) * 2)
        else:
            self.Coulomb = np.zeros((n_orbitals,) * 2)

#--------------------------------------------------------------#
#                       Main routine                           # 
#--------------------------------------------------------------#

def pprint_spin_flip_states(states):
    string = "\n"
    for state in states:
        string += str(state) + "\n"
    return string

def do(settings, molecule):

    from copy import deepcopy 
    mol = deepcopy(molecule)
    if "SF" in molecule.ExcitationType:
        make_spin_flip_states(molecule)

    molecule.States = [molecule.States[1], molecule.States[2]]

    dims = len(molecule.States)              # Dimensionality of the CI space
    CI_matrix = np.zeros((dims, dims))
    CI_overlap = np.zeros((dims, dims))
    nA = molecule.NAlphaElectrons; nB = molecule.NBetaElectrons

    # Building the CI matrix
    for i, state1 in enumerate(molecule.States):
        for j, state2 in enumerate(molecule.States[:i+1]):

            alpha = biorthogonalize(state1.Alpha.MOs, state2.Alpha.MOs, molecule.Overlap, nA, i,j)
            beta = biorthogonalize(state1.Beta.MOs, state2.Beta.MOs, molecule.Overlap, nB, i,j)

            # Calculate the core fock matrix for the state transformed into the MO basis
            alpha_core = alpha[0].T.dot(molecule.Core).dot(alpha[1]) 
            beta_core = beta[0].T.dot(molecule.Core).dot(beta[1]) 

            # Get the diagonal elements and use them to calculated the overlap, reduced overlap
            # and the list of zero overlaps
            alpha_overlaps = np.diagonal(alpha[0].T.dot(molecule.Overlap).dot(alpha[1]))
            beta_overlaps = np.diagonal(beta[0].T.dot(molecule.Overlap).dot(beta[1]))
            state_overlap = np.product(alpha_overlaps) * np.product(beta_overlaps)

            reduced_overlap, zeros_list = process_overlaps(1, [], alpha_overlaps, Spin.Alpha)
            reduced_overlap, zeros_list = process_overlaps(reduced_overlap, zeros_list, beta_overlaps, Spin.Beta)

            num_zeros = len(zeros_list)

            # Calculate the Hamiltonian matrix element for this pair of states
            # And find the combined state
            #print("Num Zeros: {} || States: {}, {}".format(num_zeros, i, j))
            
            if num_zeros == 0:
                elem,state = no_zeros(molecule, alpha, beta, alpha_overlaps, beta_overlaps, alpha_core, beta_core)
            elif num_zeros == 1:
                elem, state = one_zero(molecule, alpha, beta, alpha_overlaps, beta_overlaps, alpha_core, beta_core, zeros_list[0])
            elif num_zeros == 2:
                elem, state = two_zeros(molecule, alpha, beta, zeros_list)
            else:  # num_zeros > 2
                elem = 0

            elem *= reduced_overlap
            elem += molecule.NuclearRepulsion * state_overlap
            CI_matrix[i,j] = CI_matrix[j,i] = elem
            CI_overlap[i,j] = CI_overlap[j,i] = state_overlap

    # Print the Hamiltonian and State overlaps before attemption to solve the eigenvalue problem 
    # so we still get information if it fails
    printf.delimited_text(settings.OutFile," NOCI output ")
    printf.text_value(settings.OutFile, " Hamiltonian ", CI_matrix, " State overlaps ", CI_overlap) 

    # Solve the generalized eigenvalue problem
    #CI_matrix = np.array([[-76.766, 0.0172],[0.0172, -76.766]])
    energies, wavefunctions = gen_eig(CI_matrix, CI_overlap)

    # This is important as it allows the function to be more easily tested 
    molecule.NOCIEnergies = energies
    molecule.NOCIWavefunction = wavefunctions

    printf.text_value(settings.OutFile, " States ", wavefunctions, " NOCI Energies ", energies)

#---------------------------------------------------------------------#
#     Functions for rearranging the HF orbitals as required           #
#---------------------------------------------------------------------#
def assemble_orbitals(occupancies, optimized):
    new_MOs = np.zeros(optimized.MOs.shape)
    new_energies = np.zeros(np.shape(optimized.Energies))

    optimized_count = 0
    unoptimized_count = sum(occupancies)
    for i, occ in enumerate(occupancies):
        if occ == 0:
            new_MOs[:,unoptimized_count] = optimized.MOs[:,i]
            new_energies[unoptimized_count] = optimized.Energies[i]
            unoptimized_count += 1 
        elif occ == 1 and optimized.Occupancy[i] == 1:
            new_MOs[:,optimized_count] = optimized.MOs[:,i]
            new_energies[optimized_count] = optimized.Energies[i]
            optimized_count += 1 
        else:
            raise ValueError("Trying to construct a NOCI state without a suitable HF optimized orbital")

    return new_MOs, new_energies

def make_spin_flip_states(molecule):

    # Sort the HF states by occupancy since CUHF doesn't always order them correctly
    #for state in molecule.States:
    #    state.Alpha.sort_orbitals(molecule.Overlap, state.Total.Density)

    new_states = []
    for spin_state in molecule.SpinFlipStates:

        # Select the required HF state
        HF_state = molecule.States[spin_state[0]]
        assert HF_state.NAlpha >= HF_state.NBeta, "All high multiplicity states should have more alpha electrons than beta"

        # Now assemble the new orbitals using only the optimized alpha HF orbitals
        new_alpha, new_alpha_energies = assemble_orbitals(spin_state[1], HF_state.Alpha)
        new_beta, new_beta_energies = assemble_orbitals(spin_state[2], HF_state.Alpha)

        new_state = structures.ElectronicState(spin_state[1], spin_state[2], molecule.NOrbitals)
        new_state.Alpha.MOs = new_alpha
        new_state.Alpha.Energies = new_alpha_energies
        new_state.Beta.MOs = new_beta
        new_state.Beta.Energies = new_beta_energies
        new_state.TotalEnergy = HF_state.TotalEnergy
        new_state.Energy = HF_state.Energy

        new_states.append(new_state)

    molecule.States = new_states

#---------------------------------------------------------------------#
#  Functions for computing overlaps, densities, biorthogonalized MOs  # 
#---------------------------------------------------------------------#
def biorthogonalize(old_MOs1, old_MOs2, overlap, nElec, i, j):
    # This function finds the Lowdin Paired Orbitals for two sets of MO coefficents
    # using a singular value decomposition, as in J. Chem. Phys. 140, 114103
    # Note this only returns the MO coeffs corresponding to the occupied MOs """

    # Finding the overlap of the occupied MOs
    MOs1 = copy(old_MOs1[:,:nElec])
    MOs2 = copy(old_MOs2[:,:nElec])
    MO_overlaps = MOs1.T.dot(overlap).dot(MOs2)

    # Check if the orbitals are already paried 
    if is_biorthogonal(MOs1, MOs2, overlap):
        return [MOs1, MOs2]

    U, _sigma, Vt = np.linalg.svd(MO_overlaps)

    # Transforming each of the determinants into a biorthogonal basis
    new_MOs1 = MOs1.dot(U)
    new_MOs2 = MOs2.dot(Vt.T)
    new_overlaps = new_MOs1.T.dot(overlap).dot(new_MOs2)
     
    #assert is_biorthogonal(new_MOs1, new_MOs2, new_overlaps)

    return [new_MOs1, new_MOs2]

def is_biorthogonal(MOs1, MOs2, AO_overlaps):
    size = MOs1.shape[1]
    MO_overlaps = MOs1.T.dot(AO_overlaps).dot(MOs2)
    residuals = np.abs(MO_overlaps) - np.eye(size, size)
    biorthogonal = residuals.max() < 1e-6

    return biorthogonal

def make_weighted_density(MOs, overlaps):
    nOrbs = np.shape(MOs)[1]
    density = np.zeros((nOrbs, nOrbs))
    for i, overlap in enumerate(overlaps):
        if overlap > const.NOCI_thresh:
            P = np.outer(MOs[0][:,i], MOs[1][:,i])
            density += P / overlap
    return density

def process_overlaps(reduced_overlap, zeros_list, overlaps, spin):
    # Builds up the list of zero values as a list of (zero, spin) tuples
    # as well as the reduced overlap value
    for i, overlap in enumerate(overlaps):
        if overlap > const.NOCI_thresh:
            reduced_overlap *= overlap
        else:
            zeros_list.append(ZeroOverlap(i, spin))
    return reduced_overlap, zeros_list

def resize_array(src, dest, fill=0):
    """ Makes array src the same size as array dest, the old array is embeded in
     the upper right corner and the other elements are zero. Only works for
     for projecting a vector into a vector or a matrix into a matrix """
    old_shape = np.shape(src)
    new_array = np.full_like(dest, fill)
    if len(old_shape) is 2:              # Matrix
        height, width = old_shape
        new_array[:height, :width] = src
    else:                                # Vector
        length = old_shape[0]
        new_array[:length] = src
    return new_array

#-----------------------------------------------------------------#
#  Functions for calculating NOCI matrix elements, not including  # 
#  reduced overlap term which is accounted for later              #
#-----------------------------------------------------------------#

def no_zeros(molecule, alpha, beta, alpha_overlaps, beta_overlaps, alpha_core, beta_core):

    W_alpha = make_weighted_density(alpha, alpha_overlaps)
    W_beta = make_weighted_density(beta, beta_overlaps)
    state = CoDensityState(molecule.NOrbitals, W_alpha, W_beta)
    hf.make_coulomb_exchange_matrices(molecule, state)

    elem = inner_product(W_alpha + W_beta, state.Total.Coulomb) 
    elem += inner_product(W_alpha, state.Alpha.Exchange)
    elem += inner_product(W_beta, state.Beta.Exchange)
    elem *= 0.5

    # Add the one electron terms
    for i in range(molecule.NAlphaElectrons):
        if alpha_overlaps[i] > const.NOCI_thresh:
            elem += alpha_core[i,i] / alpha_overlaps[i]

    for i in range(molecule.NBetaElectrons):
        if beta_overlaps[i] > const.NOCI_thresh:
             elem += beta_core[i,i] / beta_overlaps[i]

    return elem, state

def one_zero(molecule, alpha, beta, alpha_overlaps, beta_overlaps, alpha_core, beta_core, zero):
    zero_index = zero.index

    # Making all the required Codensity matrices
    W_alpha = make_weighted_density(alpha, alpha_overlaps)
    W_beta = make_weighted_density(beta, beta_overlaps)
    P_alpha = np.outer(alpha[0][:,zero_index], alpha[1][:,zero_index])
    P_beta = np.outer(beta[0][:,zero_index], beta[1][:,zero_index])

    state = CoDensityState(molecule.NOrbitals, W_alpha, W_beta)
    hf.make_coulomb_exchange_matrices(molecule, state)

    active_exchange = state.Alpha.Exchange if zero.spin == Spin.Alpha else state.Beta.Exchange
    P_active = P_alpha if zero.spin == Spin.Alpha else P_beta
    active_core = alpha_core if zero.spin == Spin.Alpha else beta_core
    elem = inner_product(P_active, state.Total.Coulomb) + inner_product(P_active, active_exchange)
    elem += active_core[zero_index, zero_index]

    return elem, state

def two_zeros(molecule, alpha, beta, zeros_list):
    i, spin = zeros_list[0]

    P_alpha = np.outer(alpha[0][:,i], alpha[1][:,i])
    P_beta = np.outer(beta[0][:,i], beta[1][:,i])
    state = CoDensityState(molecule.NOrbitals, P_alpha, P_beta)
    hf.make_coulomb_exchange_matrices(molecule, state)
    active_exchange = state.Alpha.Exchange if spin == Spin.Alpha else state.Beta.Exchange
    active_P = P_alpha if spin == Spin.Alpha else P_beta

    #active_coulomb = np.zeros((molecule.NOrbitals, molecule.NOrbitals))
    #for a in range(0,molecule.NOrbitals):
    #  for b in range(0,molecule.NOrbitals):
    #    for c in range(0,molecule.NOrbitals):
    #      for d in range(0,molecule.NOrbitals):
    #         active_coulomb[a,b]  +=  active_P[c,d]*molecule.CoulombIntegrals[a,b,c,d]

    elem = inner_product(active_P, state.Total.Coulomb) + inner_product(active_P, active_exchange)
    
    return elem, state

def inner_product(mat1, mat2):
    product = mat1.dot(mat2.T)
    return np.trace(product)

def noci_pt2(molecule, settings, state):
    hf.make_fock_matrices(molecule, state)
    hf.make_MOs(molecule, state)
    mp2_correction = mp2.do(settings, molecule, [state])
    
    return mp2_correction
    
def pprint_array(arr, thresh=1e-8):
    size = len(arr)
    zero = "{:<14d}".format(0)
    line = "\n  "
    for i in range(size):
        for j in range(size):
            num = arr[i,j]
            if abs(num) > thresh:
                number = "{:<14.8f}".format(num)
            else:
                number = zero
            line += number 
        print(line)
        line = "  "
    print("")


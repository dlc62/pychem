# System libraries
import numpy as np
from scipy.linalg import eigh as gen_eig
from collections import namedtuple 
from copy import copy, deepcopy
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

    MP2_corrections = []
    if "SF" in molecule.ExcitationType:
        make_natural_orbitals(molecule)
        CI_states = [make_SF_NOCI_state(state, molecule.States) 
                     for state in molecule.SpinFlipStates]
    else:
        CI_states = molecule.States

    dims = len(CI_states)              # Dimensionality of the CI space
    CI_matrix = np.zeros((dims, dims))
    CI_overlap = np.zeros((dims, dims))
    nA = molecule.NAlphaElectrons; nB = molecule.NBetaElectrons

    # Building the CI matrix
    for i, state1 in enumerate(CI_states):
        for j, state2 in enumerate(CI_states[:i+1]):
            #print("States: {}, {}".format(i, j))

            alpha, bover = biorthogonalize(state1.Alpha.MOs, state2.Alpha.MOs, molecule.Overlap, nA)
            beta, aover = biorthogonalize(state1.Beta.MOs, state2.Beta.MOs, molecule.Overlap, nB)

            # Calculate the core fock matrix for the state transformed into the MO basis
            alpha_core = alpha[0].T.dot(molecule.Core).dot(alpha[1]) 
            beta_core = beta[0].T.dot(molecule.Core).dot(beta[1]) 

            # Get the diagonal elements and use them to calculated the overlap, reduced overlap
            # and the list of zero overlaps
            alpha_overlaps = np.diagonal(alpha[0].T.dot(molecule.Overlap).dot(alpha[1]))
            beta_overlaps = np.diagonal(beta[0].T.dot(molecule.Overlap).dot(beta[1]))
            state_overlap = np.product(alpha_overlaps) * np.product(beta_overlaps) * aover * bover

            reduced_overlap, zeros_list = process_overlaps(1, [], alpha_overlaps, Spin.Alpha)
            reduced_overlap, zeros_list = process_overlaps(reduced_overlap, zeros_list, beta_overlaps, Spin.Beta)
            reduced_overlap *= aover * bover

            num_zeros = len(zeros_list)

            # Calculate the Hamiltonian matrix element for this pair of states
            # And find the combined state
            #print("Num Zeros: {} || States: {}, {} || State Overlap: {}".format(num_zeros, i, j, state_overlap))
            
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
    energies, wavefunctions = gen_eig(CI_matrix, CI_overlap)

    molecule.NOCIEnergies = energies
    molecule.NOCIWavefunction = wavefunctions

    printf.text_value(settings.OutFile, " States ", wavefunctions, " NOCI Energies ", energies)

#---------------------------------------------------------------------#
#     Functions for rearranging the HF orbitals as required           #
#---------------------------------------------------------------------#
def assemble_orbitals(occupancies, optimized):
    new_MOs = np.empty(optimized.MOs.shape)
    new_energies = np.empty(np.shape(optimized.Energies))

    optimized_count = 0
    unoptimized_count = sum(occupancies)
    for i, occ in enumerate(occupancies):
        if occ == 0:
            new_MOs[:,unoptimized_count] = optimized.MOs[:,i]
            new_energies[unoptimized_count] = optimized.Energies[i]
            unoptimized_count += 1 
        elif occ == 1: # and optimized.Occupancy[i] == 1:
            new_MOs[:,optimized_count] = optimized.MOs[:,i]
            new_energies[optimized_count] = optimized.Energies[i]
            optimized_count += 1 
        else:
            raise ValueError("Trying to construct a NOCI state without a suitable HF optimized orbital")

    return new_MOs, new_energies

def optimize_active_space(AO_overlaps, reference, nBeta, NOs):
    # Use the doubly occupied space of the ground state as the reference orbitals 
    overlap_operator = AO_overlaps.dot(reference[:,:nBeta])
    B = NOs.T.dot(overlap_operator).dot(overlap_operator.T).dot(NOs)
    _lambda, coeffs = np.linalg.eigh(B)
    new_NOs = NOs.dot(coeffs.T)

    # Reverse the NOs to get them in order of assending occupancy
    # Probably will need to calculate the overlaps and sort in general
    new_NOs = new_NOs[:,::-1]

    return new_NOs

def make_natural_orbitals(molecule):
    reference_orbitals = None
    for i, state in enumerate(molecule.States):
        NOs, occ = hf.find_UHF_natural_orbitals(state, molecule.S)
        # Save the ground state NOs as a reference 
        if i == 0:
            reference_orbitals = NOs

        # Now we need to account for degeneracy in the singly occupied space 
        # If there is more than one singly occupied orbital we need to optimize 
        # their overlaps with the reference orbitals
        # Find the singly occupied orbitals
        idx = [i for i,x in enumerate(occ) if np.isclose(x, 0.5)]
        #idx = []
        if len(idx) > 1:
            NOs[:,idx] = optimize_active_space(
                molecule.Overlap, 
                reference_orbitals, 
                molecule.NBetaElectrons, 
                NOs[:,idx])

        state.Total.MOs = NOs
        state.Total.Energies = state.Alpha.Energies

def make_SF_NOCI_state(spin_flip_state, hf_states):
    # Select the required HF state
    NOrbs = len(spin_flip_state[1])
    HF_state = hf_states[spin_flip_state[0]]

    # Now assemble the new orbitals using only the optimized alpha HF orbitals
    new_alpha, new_alpha_energies = assemble_orbitals(spin_flip_state[1], HF_state.Total)
    new_beta, new_beta_energies = assemble_orbitals(spin_flip_state[2], HF_state.Total)

    new_state = structures.ElectronicState(spin_flip_state[1], spin_flip_state[2], NOrbs)
    new_state.Alpha.MOs = new_alpha
    new_state.Alpha.Energies = new_alpha_energies
    new_state.Beta.MOs = new_beta
    new_state.Beta.Energies = new_beta_energies
    new_state.TotalEnergy = HF_state.TotalEnergy
    new_state.Energy = HF_state.Energy

    return new_state

#---------------------------------------------------------------------#
#  Functions for computing overlaps, densities, biorthogonalized MOs  # 
#---------------------------------------------------------------------#
def biorthogonalize(old_MOs1, old_MOs2, AO_overlaps, nElec):
    # This function finds the Lowdin Paired Orbitals for two sets of MO coefficents
    # using a singular value decomposition, as in J. Chem. Phys. 140, 114103
    # Note this only returns the MO coeffs corresponding to the occupied MOs """

    # Finding the overlap of the occupied MOs
    MOs1 = copy(old_MOs1[:,:nElec])
    MOs2 = copy(old_MOs2[:,:nElec])
    MO_overlaps = MOs1.T.dot(AO_overlaps).dot(MOs2)

    # Check if the orbitals are already paried 
    if is_biorthogonal(MOs1, MOs2, AO_overlaps):
        return [MOs1, MOs2], 1

    U, _sigma, Vt = np.linalg.svd(MO_overlaps)

    over = np.linalg.det(U) * np.linalg.det(Vt)

    # Transforming each of the determinants into a biorthogonal basis
    new_MOs1 = MOs1.dot(U) 
    new_MOs2 = MOs2.dot(Vt.T) 

    # Leave these variables for debugging for now 
    new_overlaps = new_MOs1.T.dot(AO_overlaps).dot(new_MOs2)
    old_det = np.linalg.det(MO_overlaps)
    new_det = np.linalg.det(new_overlaps)

    assert is_biorthogonal(new_MOs1, new_MOs2, AO_overlaps)

    return [new_MOs1, new_MOs2], over

def is_biorthogonal(MOs1, MOs2, AO_overlaps):
    size = MOs1.shape[1]
    MO_overlaps = MOs1.T.dot(AO_overlaps).dot(MOs2)
    residuals = np.abs(MO_overlaps) - np.eye(size, size)
    biorthogonal = residuals.max() < const.NOCI_thresh

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
        if np.abs(overlap) > const.NOCI_thresh:
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

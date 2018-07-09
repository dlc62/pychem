# System libraries
import copy
import numpy as np
from scipy.linalg import eigh as gen_eig
# Custom code
from hartree_fock import make_coulomb_exchange_matrices, make_density_matrices
from Util import printf
from Data import constants as const
from Util import structures

#--------------------------------------------------------------#
#                Set up required structures                    # 
#--------------------------------------------------------------#
class CoDensityState:
    # A state-like object used for calculating and storing the pseudo Fock matrix
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
def do(settings, molecule):

    if "SF" in molecule.ExcitationType:
        reorder_orbitals(molecule)

    dims = len(molecule.States)              # Dimensionality of the CI space
    CI_matrix = np.zeros((dims, dims))
    CI_overlap = np.zeros((dims, dims))
    nA = molecule.NAlphaElectrons; nB = molecule.NBetaElectrons

    # Building the CI matrix
    for i, state1 in enumerate(molecule.States):
        for j, state2 in enumerate(molecule.States[:i+1]):

            # Biorthogonalize the occupied MOs
            alpha = biorthogonalize(state1.Alpha.MOs[:,0:nA], state2.Alpha.MOs[:,0:nA], molecule.Overlap)
            beta = biorthogonalize(state1.Beta.MOs[:,0:nB], state2.Beta.MOs[:,0:nB], molecule.Overlap)


            # Calculate the core fock matrix for the state transformed into the MO basis
            alpha_core = alpha[0].T.dot(molecule.Core).dot(alpha[1])
            beta_core = beta[0].T.dot(molecule.Core).dot(beta[1])

            # Get the diagonal elements and use them to calculated the overlap, reduced overlap
            # and the list of zero overlaps
            alpha_overlaps = np.diagonal(alpha[0].T.dot(molecule.Overlap).dot(alpha[1]))
            beta_overlaps = np.diagonal(beta[0].T.dot(molecule.Overlap).dot(beta[1]))
            state_overlap = (np.product(alpha_overlaps) * np.product(beta_overlaps))


            #state_overlap = (np.product(alpha_overlaps) + np.product(beta_overlaps)) / 2
            reduced_overlap, zeros_list = process_overlaps(1, [], alpha_overlaps, "alpha")
            reduced_overlap, zeros_list = process_overlaps(reduced_overlap, zeros_list, beta_overlaps, "beta")

            # Ensure that the alpha and beta arrays are the same size
            if nA > nB:
                beta_overlaps = resize_array(beta_overlaps, alpha_overlaps, fill=1)   # Fill with 1s so as not to affect the product
                beta[0] = resize_array(beta[0], alpha[0])
                beta[1] = resize_array(beta[1], alpha[1])

            num_zeros = len(zeros_list)

        # Calculate the Hamiltonian matrix element for this pair of states
            #print("Num Zeros: {} || States: {}, {}".format(num_zeros, i, j))
            if num_zeros is 0:
                elem = no_zeros(molecule, alpha, beta, alpha_overlaps, beta_overlaps, alpha_core, beta_core)
            elif num_zeros is 1:
                elem = one_zero(molecule, alpha, beta, alpha_overlaps, beta_overlaps, alpha_core, beta_core, zeros_list[0])
            elif num_zeros is 2:
                elem = two_zeros(molecule, alpha, beta, zeros_list)
            else:  # num_zeros > 2
                elem = 0
            elem *= reduced_overlap
            elem += molecule.NuclearRepulsion * state_overlap

            CI_matrix[i,j] = CI_matrix[j,i] = elem
            CI_overlap[i,j] = CI_overlap[j,i] = state_overlap

    # Solve the generalized eigenvalue problem
    energies, wavefunctions = gen_eig(CI_matrix, CI_overlap)
    molecule.NOCIEnergies = energies
    molecule.NOCIWavefunction = wavefunctions

    # Print the results to file
    printf.delimited_text(settings.OutFile," NOCI output ")
    printf.text_value(settings.OutFile, " States ", wavefunctions, " NOCI Energies ", energies)
    if settings.PrintLevel == "VERBOSE" or settings.PrintLevel == "DEBUG":
       printf.text_value(settings.OutFile, " Hamiltonian ", CI_matrix, " State overlaps ", CI_overlap) 

#---------------------------------------------------------------------#
#     Functions for rearranging the HF orbitals as required           #
#---------------------------------------------------------------------#
def assemble_orbitals(occupancies, optimized):
    new_MOs = np.zeros(optimized.MOs.shape)

    n_orbitals_occupied = 0
    for i, occ in enumerate(occupancies):
        if occ == 0:
            continue
        elif occ == 1 and optimized.Occupancy[i] == 1:
            new_MOs[:,n_orbitals_occupied] = optimized.MOs[:,i]
            n_orbitals_occupied += 1 
        else:
            raise ValueError("Trying to construct a NOCI state without a suitable HF optimized orbital")

    return new_MOs


def reorder_orbitals(molecule):

    new_states = []
    for spin_state in molecule.SpinFlipStates:

        # Select the required HF state
        HF_state = molecule.States[spin_state[0]]
        assert HF_state.NAlpha >= HF_state.NBeta, "All high multiplicity states should have more alpha electrons than beta"

        # Now assemble the new orbitals using only the optimized alpha HF orbitals
        new_alpha = assemble_orbitals(spin_state[1], HF_state.Alpha)
        new_beta = assemble_orbitals(spin_state[2], HF_state.Alpha)

        new_state = structures.ElectronicState(spin_state[1], spin_state[2], molecule.NOrbitals)
        new_state.Alpha.MOs = new_alpha
        new_state.Beta.MOs = new_beta
        new_state.TotalEnergy = HF_state.TotalEnergy
        new_state.Energy = HF_state.Energy
        make_density_matrices(molecule, new_state)

        new_states.append(new_state)

    if new_states != []:
        molecule.States = new_states


#---------------------------------------------------------------------#
#  Functions for computing overlaps, densities, biorthogonalized MOs  # 
#---------------------------------------------------------------------#
def biorthogonalize(MOs1, MOs2, overlap):
    # This function finds the Lowdin Paired Orbitals for two sets of MO coefficents
    # using a singular value decomposition, as in J. Chem. Phys. 140, 114103
    # Note this only returns the MO coeffs corresponding to the occupied MOs """

    # Finding the overlap of the occupied MOs
    det_overlap = MOs1.T.dot(overlap).dot(MOs2)

    U, _, Vt = np.linalg.svd(det_overlap)

    # Transforming each of the determinants into a biorthogonal basis
    new_MOs1 = MOs1.dot(U)
    new_MOs2 = MOs2.dot(Vt.T)

    return [new_MOs1, new_MOs2]

def make_weighted_density(MOs, overlaps):
    nOrbs = np.shape(MOs)[1]
    density = np.zeros((nOrbs, nOrbs))
    for i, overlap in enumerate(overlaps):
        if overlap > const.NOCI_thresh:
            P = np.outer(MOs[0][:,i], MOs[1][:,i])
            density += P / overlap
        else:
             density += np.outer(MOs[0][:,i], MOs[1][:,i])
    return density

def process_overlaps(reduced_overlap, zeros_list, overlaps, spin):
    # Builds up the list of zero values as a list of (zero, spin) tuples
    # as well as the reduced overlap value
    for i, overlap in enumerate(overlaps):
        if overlap > const.NOCI_thresh:
            reduced_overlap *= overlap
        else:
            zeros_list.append((i,spin))
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
    make_coulomb_exchange_matrices(molecule, state)

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

    return elem

def one_zero(molecule, alpha, beta, alpha_overlaps, beta_overlaps, alpha_core, beta_core, zero):

    zero_index = zero[0]

    # Making all the required Codensity matrices
    W_alpha = make_weighted_density(alpha, alpha_overlaps)
    W_beta = make_weighted_density(beta, beta_overlaps)
    P_alpha = np.outer(alpha[0][:,zero_index], alpha[1][:,zero_index])
    P_beta = np.outer(beta[0][:,zero_index], beta[1][:,zero_index])

    state = CoDensityState(molecule.NOrbitals, W_alpha, W_beta)
    make_coulomb_exchange_matrices(molecule, state)

    active_exchange = state.Alpha.Exchange if zero[1] == "alpha" else state.Beta.Exchange
    P_active = P_alpha if zero[1] == "alpha" else P_beta
    active_core = alpha_core if zero[1] == "alpha" else beta_core

    elem = inner_product(P_active, state.Total.Coulomb) + inner_product(P_active, active_exchange)
    elem += active_core[zero_index, zero_index]

    return elem

def two_zeros(molecule, alpha, beta, zeros_list):
    i = zeros_list[0][0]
    spin = zeros_list[0][1]

    P_alpha = np.outer(alpha[0][:,i], alpha[1][:,i])
    P_beta = np.outer(beta[0][:,i], beta[1][:,i])
    state = CoDensityState(molecule.NOrbitals, P_alpha, P_beta)
    make_coulomb_exchange_matrices(molecule, state)
    active_exhange = state.Alpha.Exchange if spin == 'alpha' else state.Beta.Exchange
    active_P = P_alpha if spin == "alpha" else P_beta
    elem = inner_product(active_P, state.Total.Coulomb) + inner_product(active_P, active_exhange)

    return elem

def inner_product(mat1, mat2):
    product = mat1.dot(mat2.T)
    return np.trace(product)

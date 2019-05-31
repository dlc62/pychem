# System libraries
import numpy as np
from scipy.linalg import eigh as gen_eig
from collections import namedtuple 
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

def pprint_spin_flip_states(states):
    string = "\n"
    for state in states:
        string += str(state) + "\n"
    return string

def do(settings, molecule):

    if "SF" in molecule.ExcitationType:
        reorder_orbitals(molecule)

    MP2_corrections = []
    dims = len(molecule.States)              # Dimensionality of the CI space
    CI_matrix = np.zeros((dims, dims))
    CI_overlap = np.zeros((dims, dims))
    nA = molecule.NAlphaElectrons; nB = molecule.NBetaElectrons

    # Building the CI matrix
    for i, state1 in enumerate(molecule.States):
        for j, state2 in enumerate(molecule.States[:i+1]):

            alpha = biorthogonalize(state1.Alpha.MOs, state2.Alpha.MOs, molecule.Overlap, nA)
            beta = biorthogonalize(state1.Beta.MOs, state2.Beta.MOs, molecule.Overlap, nB)

            # Calculate the core fock matrix for the state transformed into the MO basis
            alpha_core = alpha[0].T.dot(molecule.Core).dot(alpha[1])
            beta_core = beta[0].T.dot(molecule.Core).dot(beta[1])

            # Get the diagonal elements and use them to calculated the overlap, reduced overlap
            # and the list of zero overlaps
            alpha_overlaps = np.diagonal(alpha[0].T.dot(molecule.Overlap).dot(alpha[1]))
            beta_overlaps = np.diagonal(beta[0].T.dot(molecule.Overlap).dot(beta[1]))
            state_overlap = (np.product(alpha_overlaps) * np.product(beta_overlaps))

            #state_overlap = (np.product(alpha_overlaps) + np.product(beta_overlaps)) / 2
            reduced_overlap, zeros_list = process_overlaps(1, [], alpha_overlaps, Spin.Alpha)
            reduced_overlap, zeros_list = process_overlaps(reduced_overlap, zeros_list, beta_overlaps, Spin.Beta)

            # Ensure that the alpha and beta arrays are the same size
            if nA > nB:
                beta_overlaps = resize_array(beta_overlaps, alpha_overlaps, fill=1)   # Fill with 1s so as not to affect the product
                beta[0] = resize_array(beta[0], alpha[0])
                beta[1] = resize_array(beta[1], alpha[1])

            num_zeros = len(zeros_list)

            # Calculate the Hamiltonian matrix element for this pair of states
            # And find the combined state
            # print("Num Zeros: {} || States: {}, {}".format(num_zeros, i, j))
            if num_zeros is 0:
                elem,state = no_zeros(molecule, alpha, beta, alpha_overlaps, beta_overlaps, alpha_core, beta_core)
            elif num_zeros is 1:
                elem, state = one_zero(molecule, alpha, beta, alpha_overlaps, beta_overlaps, alpha_core, beta_core, zeros_list[0])
            elif num_zeros is 2:
                elem, state = two_zeros(molecule, alpha, beta, zeros_list)
            else:  # num_zeros > 2
                elem = 0

            elem *= reduced_overlap
            elem += molecule.NuclearRepulsion * state_overlap
            CI_matrix[i,j] = CI_matrix[j,i] = elem
            CI_overlap[i,j] = CI_overlap[j,i] = state_overlap

    # Solve the generalized eigenvalue problem
    energies, wavefunctions = gen_eig(CI_matrix, CI_overlap)

    # TODO find a better way to do this 
    #if settings.MP2_type == "AFTER":
    #    for i in range(len(energies)):
    #        for j, coeff in enumerate(wavefunctions[:,i]):
    #            energies[i] += MP2_corrections[j] * coeff**2

    molecule.NOCIEnergies = energies
    molecule.NOCIWavefunction = wavefunctions


    # Print the results to file
    printf.delimited_text(settings.OutFile," NOCI output ")

    # printf.text_value(settings.OutFile, " Spin Flip States ", pprint_spin_flip_states(molecule.SpinFlipStates))

    printf.text_value(settings.OutFile, " States ", wavefunctions, " NOCI Energies ", energies)
    if settings.PrintLevel == "VERBOSE" or settings.PrintLevel == "DEBUG" or True:
       printf.text_value(settings.OutFile, " Hamiltonian ", CI_matrix, " State overlaps ", CI_overlap) 

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


def reorder_orbitals(molecule):

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
        hf.make_density_matrices(molecule, new_state)

        new_states.append(new_state)

    if new_states != []:
        molecule.States = new_states

#
#---------------------------------------------------------------------#
#  Functions for computing overlaps, densities, biorthogonalized MOs  # 
#---------------------------------------------------------------------#
def biorthogonalize(MOs1, MOs2, overlap, nElec):
    # This function finds the Lowdin Paired Orbitals for two sets of MO coefficents
    # using a singular value decomposition, as in J. Chem. Phys. 140, 114103
    # Note this only returns the MO coeffs corresponding to the occupied MOs """

    # Finding the overlap of the occupied MOs
    MOs1 = MOs1[:,:nElec]
    MOs2 = MOs2[:,:nElec]
    det_overlap = MOs1.T.dot(overlap).dot(MOs2)

    U, _, Vt = np.linalg.svd(det_overlap)

    # Transforming each of the determinants into a biorthogonal basis
    new_MOs1 = MOs1.dot(U)
    new_MOs2 = MOs2.dot(Vt.T)

    # Ensure the new orbitals are in phase 
    overlaps = [orb1.dot(orb2) for orb1, orb2 in zip(new_MOs1.T, new_MOs2.T)]
    for i, overlap in enumerate(overlaps):
        if overlap < 0:
            new_MOs2[:,i] *= -1 

    return [new_MOs1, new_MOs2]

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

    active_coulomb = np.zeros((molecule.NOrbitals, molecule.NOrbitals))
    for a in range(0,molecule.NOrbitals):
      for b in range(0,molecule.NOrbitals):
        for c in range(0,molecule.NOrbitals):
          for d in range(0,molecule.NOrbitals):
             active_coulomb[a,b]  +=  active_P[c,d]*molecule.CoulombIntegrals[a,b,c,d]

    elem = inner_product(active_P, state.Total.Coulomb) + inner_product(active_P, active_exchange)

    return elem, state

def set_to_zeros(array):
    array = np.zeros(array.shape)

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

def extend_biorthogonalize(orbs, molecule):
    # Takes the set of biorthoginalized occupied orbitals and finds the 
    # corresponding paired virtual orbitals
    # Assumes the same number of electrons in both sets
    # See Int. J. Quantum. Chem. 29, 31 (1986)
    nOrbs = molecule.NOrbitals
    nElec = orbs[0].shape[1]
    pair1 = np.zeros((nOrbs, nOrbs))
    pair2 = np.zeros((nOrbs, nOrbs))

    pair1[:, 0:nElec] = orbs[0]
    pair2[:, 0:nElec] = orbs[1]

    # Calculate the first count virtual orbitals
    # Idealy this will find nElec orbitals but in cases where overlap is 1, 
    # it will be unable to find one 
    # TODO think about overflows they're definitly going to be a problem
    count = 0 
    stop = min(nElec, nOrbs - nElec)
    for i in range(stop): 
        j = nElec + count 
        overlap = pair1[:,i].dot(molecule.Overlap).dot(pair2[:,i])   
        norm = np.sqrt(1-overlap**2) if abs(overlap) < 1 else 0
        
        if norm > 1e-5: 
            pair1[:,j] = norm * (pair2[:,i] - overlap * pair1[:,i])
            pair2[:,j] = norm * (pair1[:,i] - overlap * pair2[:,i])  # Assuming overlap is real 
            count += 1 
        else:
            continue 

    # Find the remaining nOrbs - nElec - count orbitals
    span = pair1.T.dot(molecule.Overlap)
    import pdb; pdb.set_trace()
    perp = np.linalg.null(span)

    pair1[:,(2*nElec):] = perp
    pair2[:,(2*nElec):] = perp

    return pair1, pair2 


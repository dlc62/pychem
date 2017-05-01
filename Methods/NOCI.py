# Import system libraries
import numpy as np
from scipy.linalg import eigh as gen_eig
import copy

#import pychem modules
from Methods import integrals
from Data.constants import NOCI_Thresh as THRESH
from Util.util import inner_product, resize_array, occupied
from Util import printf
from Methods.mom import make_p_vector, Sort_MOs

def do_NOCI(molecule, settings):
    """ Main function - takes a molecule and perfroms NOCI using all the availible
        States """
    dims = len(molecule.States)              # Dimensionality of the CI space
    CI_matrix = np.zeros((dims, dims))
    CI_overlap = np.zeros((dims, dims))

    # Building the CI matrix
    for i, state1 in enumerate(molecule.States):
        for j, state2 in enumerate(molecule.States[:i+1]):

            # For the diagonal elements the energy is just
            # the state energy and the overlap is 1
            if i == j:
                CI_matrix[i,i] = state1.TotalEnergy
                CI_overlap[i,i] = 1
                continue

            # Biorthoginalize the occupied MOs
            alpha = biorthoginalize(state1.Alpha, state2.Alpha, molecule)
            beta = biorthoginalize(state1.Beta, state2.Beta, molecule)

            #alpha[0] = match_orbitals(alpha[0], alpha[1], molecule.NAlphaElectrons, molecule.Overlap)
            #beta[0] = match_orbitals(beta[0], beta[1], molecule.NBetaElectrons, molecule.Overlap)

            # Calculate the core fock matrix for the state transformed into the MOs basis
            molecule.alpha_core = alpha[0].T.dot(molecule.Core).dot(alpha[1])
            molecule.beta_core = beta[0].T.dot(molecule.Core).dot(beta[1])

            # Get the diagonal elements and use them to calculated the overlap, reduced overlap
            # and the list of zero overlaps
            alpha_overlaps = np.diagonal(alpha[0].T.dot(molecule.Overlap).dot(alpha[1]))
            beta_overlaps = np.diagonal(beta[0].T.dot(molecule.Overlap).dot(beta[1]))
            state_overlap = (np.product(alpha_overlaps) * np.product(beta_overlaps))

            #state_overlap = (np.product(alpha_overlaps) + np.product(beta_overlaps)) / 2
            reduced_overlap, zeros_list = process_overlaps(1, [], alpha_overlaps, "alpha")
            reduced_overlap, zeros_list = process_overlaps(reduced_overlap, zeros_list, beta_overlaps, "beta")

            # Ensure that the alpha and beta arrays are the same size
            if molecule.NAlphaElectrons > molecule.NBetaElectrons:
                beta_overlaps = resize_array(beta_overlaps, alpha_overlaps, fill=1)   # Fill with 1s so as not to effect the product
                beta[0] = resize_array(beta[0], alpha[0])
                beta[1] = resize_array(beta[1], alpha[1])

            num_zeros = len(zeros_list)

        # Calculated the element of the Hamiltonian matrix
            #print("Num Zeros: {} || States: {}, {}".format(num_zeros, i, j))
            if num_zeros is 0:
                elem = no_zeros(alpha, beta, alpha_overlaps, beta_overlaps, molecule)
            elif num_zeros is 1:
                elem = one_zero(alpha, beta, alpha_overlaps, beta_overlaps, zeros_list[0], molecule)
            elif num_zeros is 2:
                elem = two_zeros(alpha, beta, zeros_list, molecule)
            else:  # num_zeros > 2
                elem = 0
            elem *= reduced_overlap
            elem += molecule.NuclearRepulsion * state_overlap

            CI_matrix[i,j] = CI_matrix[j,i] = elem
            CI_overlap[i,j] = CI_overlap[j,i] = state_overlap

    # Solve the generalized eigenvalue problem
    try:
        energies, wavefunctions = gen_eig(CI_matrix, CI_overlap)
    except np.linalg.linalg.LinAlgError:
        printf.printf(settings, """Could not solve NOCI equations, this suggests two MOM calculations converged to the same state""")
        energies = "ERROR"
        wavefunctions = "ERROR"

    # Check the CI wavefunction is properly normalized
    try:
        assert np.allclose(wavefunctions.T.dot(CI_overlap).dot(wavefunctions), np.eye(dims))
    except:
        print("CI WAVEFUNCTION NOT NORMALIZED")

    printf.NOCI(settings, CI_matrix, CI_overlap, wavefunctions, energies)

def biorthoginalize(state1, state2, molecule):
    # This function finds the Lowdin Paired Orbitals for two sets of MO coefficents
    # using a singular value decomposition, as in J. Chem. Phys. 140, 114103
    # Note this only returns the MO coeffs corresponding to the occupied MOs """

    # Finding the overlap of the occupied MOs
    MOs1 = occupied(state1);     MOs2 = occupied(state2)
    det_overlap = MOs1.T.dot(molecule.Overlap).dot(MOs2)

    U, _, Vt = np.linalg.svd(det_overlap)

    # Transforming each of the determinants into a biorthoginal basis
    new_MOs1 = MOs1.dot(U.T)
    new_MOs2 = MOs2.dot(Vt)

    # ensuring the relative phases of the orbitals are maintained
    MO1_overlap = new_MOs1[:,0].dot(molecule.Overlap).dot(new_MOs2[:,0])
    if MO1_overlap < 0:
        new_MOs1 *= -1

    return [new_MOs1, new_MOs2]      # This is returned as a list to allows reassignment later

class CoDensity_State:
    # A state like object used for calculating and
    # storing the psudo fock matrix
    def __init__(self, Alpha_Density, Beta_Density):
        size = len(Alpha_Density)
        self.alpha = Alpha_Density
        self.beta = Beta_Density
        self.total = Alpha_Density + Beta_Density
        self.coulomb = np.zeros((size, size))
        self.alpha_exchange = np.zeros((size, size))
        self.beta_exchange = np.zeros((size, size))

def make_coulomb_exchange_matrices(molecule, state):
    # Modified version for calculating the coulomb and exhange matrices without
    #building the fock matrices

    # Check if the ColoumbIntegrals have been stored in core
    if hasattr(molecule, "ColoumbIntegrals"):
        ints_handling = "INCORE"
    else:
        ints_handling = "DIRECT"

    for shell_pair1 in molecule.ShellPairs:
        ia_vec = shell_pair1.Centre1.Ivec
        ib_vec = shell_pair1.Centre2.Ivec
        for shell_pair2 in molecule.ShellPairs:
            ic_vec = shell_pair2.Centre1.Ivec
            id_vec = shell_pair2.Centre2.Ivec

            # Calculate integrals if direct HF or first iteration
            if (ints_handling == 'DIRECT'):
                coulomb,exchange = integrals.two_electron(shell_pair1,shell_pair2)

            for m in range(0,shell_pair1.Centre1.Cgtf.NAngMom):
                for n in range(0,shell_pair1.Centre2.Cgtf.NAngMom):
                    for l in range(0,shell_pair2.Centre1.Cgtf.NAngMom):
                        for s in range(0,shell_pair2.Centre2.Cgtf.NAngMom):

                            # Construct coulomb and exchange matrices
                            if (ints_handling == 'INCORE'):
                                state.coulomb[ia_vec[m]][ib_vec[n]]  +=  state.total[ic_vec[l]][id_vec[s]]* \
                                                                              molecule.CoulombIntegrals[ia_vec[m],ib_vec[n],ic_vec[l],id_vec[s]]
                                state.alpha_exchange[ia_vec[m]][ib_vec[n]] += -state.alpha[ic_vec[l]][id_vec[s]]* \
                                                                              molecule.ExchangeIntegrals[ia_vec[m],id_vec[s],ic_vec[l],ib_vec[n]]
                                state.beta_exchange[ia_vec[m]][ib_vec[n]]  += -state.beta[ic_vec[l]][id_vec[s]]* \
                                                                              molecule.ExchangeIntegrals[ia_vec[m],id_vec[s],ic_vec[l],ib_vec[n]]
                            else:
                                state.coulomb[ia_vec[m]][ib_vec[n]]  +=  state.total[ic_vec[l]][id_vec[s]]*coulomb[m][n][l][s]
                                state.alpha_exchange[ia_vec[m]][ib_vec[n]] += -state.alpha[ic_vec[l]][id_vec[s]]*exchange[m][s][l][n]
                                state.beta_exchange[ia_vec[m]][ib_vec[n]]  += -state.beta[ic_vec[l]][id_vec[s]]*exchange[m][s][l][n]

def make_weighted_density(MOs, overlaps, molecule):
    nOrbs = np.shape(MOs)[1]
    density = np.zeros((nOrbs, nOrbs))
    for i, overlap in enumerate(overlaps):
        if overlap > THRESH:
            P = np.outer(MOs[0][:,i], MOs[1][:,i])
            density += P / overlap
        else:
             density += np.outer(MOs[0][:,i], MOs[1][:,i])
    return density

def process_overlaps(reduced_overlap, zeros_list, overlaps, spin):
    # Builds up the list of zero values as a list of (zero, spin) tuples
    # as well as the reduced overlap value
    for i, overlap in enumerate(overlaps):
        if overlap > THRESH:
            reduced_overlap *= overlap
        else:
            zeros_list.append((i,spin))
    return reduced_overlap, zeros_list

# -----------------------------------------------------------#
#     Functions for calculating the CI matrix elements       #
#       Note: these do not include the reduced overlap       #
#             term, that is included latter.                 #
# -----------------------------------------------------------#

def no_zeros(alpha, beta, alpha_overlaps, beta_overlaps, molecule):

    W_alpha = make_weighted_density(alpha, alpha_overlaps, molecule)
    W_beta = make_weighted_density(beta, beta_overlaps, molecule)
    state = CoDensity_State(W_alpha, W_beta)
    make_coulomb_exchange_matrices(molecule, state)

    elem = inner_product(W_alpha + W_beta, state.coulomb)
    elem += inner_product(W_alpha, state.alpha_exchange)
    elem += inner_product(W_beta, state.beta_exchange)
    elem *= 0.5

    # Add the one electron terms
    for i in range(molecule.NAlphaElectrons):
        if alpha_overlaps[i] > THRESH:
            elem += molecule.alpha_core[i,i] / alpha_overlaps[i]

    for i in range(molecule.NBetaElectrons):
        if beta_overlaps[i] > THRESH:
             elem += molecule.beta_core[i,i] / beta_overlaps[i]

    return elem

def one_zero(alpha, beta, alpha_overlaps, beta_overlaps, zero, molecule):
    zero_index = zero[0]

    # Making all the required Codensity matrices
    W_alpha = make_weighted_density(alpha, alpha_overlaps, molecule)
    W_beta = make_weighted_density(beta, beta_overlaps, molecule)
    P_alpha = np.outer(alpha[0][:,zero_index], alpha[1][:,zero_index])
    P_beta = np.outer(beta[0][:,zero_index], beta[1][:,zero_index])

    state = CoDensity_State(W_alpha, W_beta)
    make_coulomb_exchange_matrices(molecule, state)

    active_exchange = state.alpha_exchange if zero[1] == "alpha" else state.beta_exchange
    P_active = P_alpha if zero[1] == "alpha" else P_beta
    active_core = molecule.alpha_core if zero[1] == "alpha" else molecule.beta_core

    elem = inner_product(P_active, state.coulomb) + inner_product(P_active, active_exchange)
    elem += active_core[zero_index, zero_index]

    return elem

def two_zeros(alpha, beta, zeros_list, molecule):
    elem = 0

    # This actually seems to only need one of the zeros
    i = zeros_list[0][0]
    spin = zeros_list[0][1]

    P_alpha = np.outer(alpha[0][:,i], alpha[1][:,i])
    P_beta = np.outer(beta[0][:,i], beta[1][:,i])
    state = CoDensity_State(P_alpha, P_beta)
    make_coulomb_exchange_matrices(molecule, state)
    active_exhange = state.alpha_exchange if spin == 'alpha' else state.beta_exchange
    active_P = P_alpha if spin == "alpha" else P_beta
    elem += inner_product(active_P, state.coulomb) + inner_product(active_P, active_exhange)

    return elem

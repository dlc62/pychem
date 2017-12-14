# System libraries
from __future__ import print_function
import numpy
from numpy import dot
import scipy
from scipy.linalg import sqrtm
import copy
from math import floor

# Custom-written data modules
import Data.constants as c

# Custom-written code modules
from Util import util, printf
from Methods import integrals, diis, mom

#=================================================================#
#                        MAIN SUBROUTINE                          #
#=================================================================#

def do_SCF(settings, molecule, state_index = 0):
    state = copy.deepcopy(molecule.States[state_index])
    # Check if this state is just the result of swapping the spin labels from a
    # previouly calculated state and if so return

    is_swap, state = check_swapped_spin(molecule, state, state_index, settings)
    if is_swap:
        molecule.States[state_index] = state
        return 0

    # Calculate values that are constant throughout the calculation
    molecule.NuclearRepulsion = integrals.nuclear_repulsion(molecule)
    make_core_matrices(molecule)

    # Permute the occupancy and set new number of alpha and beta
    # Electrons if doing a spin flip calculation
    if settings.SCF.spin_flip:
        do_spin_flip(molecule,state)

    # Set up for SCF calculation
    initialize_fock_matrices(molecule.Core, state)
    reference_orbitals = load_reference_orbitals(settings, state)
    init_density_and_MOs(state, molecule, settings, state_index)

    # Calculate initial energy
    calculate_energy(molecule, state)
    state.dE = state.Energy
    energies = []

    # Print initial
    printf.HF_Initial(molecule, state, settings)

    #-------------------------------------------#
    #           Begin SCF Iterations            #
    #-------------------------------------------#
    num_iterations = 0
    final_loop = False
    diis_error = None

    if molecule.Basis == settings.BasisSets[-1]:
        energy_convergence = c.energy_convergence_final
    else:
        energy_convergence = c.energy_convergence

    #while energy_convergence < abs(state.dE):
    while not final_loop:
        num_iterations += 1

        #-------------------------------------------#
        #               Main SCF step               #
        #-------------------------------------------#
        initialize_fock_matrices(molecule.Core, state)

        # Average the density marices for RHF
        if settings.SCF.Reference == "RHF":
            state.Alpha.Density = numpy.mean([state.Alpha.Density, state.Beta.Density], axis=0)
            state.Beta.Density = copy.copy(state.Alpha.Density)

        make_coulomb_exchange_matrices(molecule, state, settings.SCF.Ints_Handling, num_iterations)

        #-------------------------------------------#
        #    Convergence accelerators/modifiers     #
        #-------------------------------------------#

        # DIIS
        if settings.DIIS.Use:
            if state_index != 0 or not settings.SCF.spin_flip:
                settings.DIIS.Damp = True
            diis.do(molecule, state, settings)
            diis_error = max(state.AlphaDIIS.Error, state.BetaDIIS.Error)

        if settings.SCF.Reference == "CUHF":
            constrain_UHF(molecule, state, state_index)

        make_MOs(molecule, state)

        calc_spin(molecule, state)

        # Optionally, use MOM to reorder MOs
        if settings.MOM.Use and reference_orbitals:
            mom.do(molecule, state, state_index, reference_orbitals)
            if settings.MOM.Reference == 'MUTABLE':
                reference_orbitals = [state.Alpha.MOs, state.Beta.MOs]
        #-------------------------------------------#

        # Sort the occupied MOs by energy
        util.sort_MOs(state.Alpha, molecule)
        util.sort_MOs(state.Beta, molecule)
        make_density_matrices(molecule,state)

        old_energy = state.Energy
        calculate_energy(molecule, state)
        state.dE = state.Energy - old_energy
        state.TotalEnergy = state.Energy + molecule.NuclearRepulsion
        energies.append(state.TotalEnergy)

        if abs(state.dE) < energy_convergence or num_iterations >= settings.SCF.MaxIter:
            final_loop = True
            if settings.SCF.spin_flip:
                undo_spin_flip(molecule, state, state_index)

        printf.HF_Loop(state, settings, num_iterations, diis_error, final_loop)

        if num_iterations >= settings.SCF.MaxIter:
            print("SCF not converging")
            break

    molecule.States[state_index] = state
    printf.HF_Final(settings)
#---------------------------------------------------------------------------#
#            Basic HF subroutines, this = this electronic state             #
#---------------------------------------------------------------------------#

def initialize_fock_matrices(core,this):

    this.Alpha.Fock = copy.deepcopy(core)
    this.Beta.Fock = copy.deepcopy(core)

#----------------------------------------------------------------------

def make_MOs(molecule,this):

    X = molecule.X
    Xt = molecule.Xt
    this.Alpha.Energies,Ca = numpy.linalg.eigh(Xt.dot(this.Alpha.Fock).dot(X))
    this.Beta.Energies,Cb = numpy.linalg.eigh(Xt.dot(this.Beta.Fock).dot(X))
    this.Alpha.MOs = numpy.dot(X,Ca)
    this.Beta.MOs = numpy.dot(X,Cb)

#----------------------------------------------------------------------

def load_reference_orbitals(settings, state):
    reference_exists = state.Alpha.MOs != []
    if reference_exists and not settings.SCF.spin_flip:
        return [state.Alpha.MOs, state.Beta.MOs]
    else:
        return None

#----------------------------------------------------------------------

def init_density_and_MOs(state, molecule, settings, state_index):
    # Read MOs from file
    if settings.SCF.Guess == "READ" and molecule.Basis == settings.BasisSets[0]:
        try:
            states = util.fetch("MOs", settings.SCF.MOReadName, settings.SCF.MOReadBasis)
            state = states[state_index]
            assert len(state.Alpha.MOs) == molecule.NOrbitals
            diis.reset_diis(state.AlphaDIIS)    # Throwing out diis information from the
            diis.reset_diis(state.BetaDIIS)     # previous calculation
        except:
            print("Could not read MOs from file")
            import sys; sys.exit()
    # Make completly new MOs
    elif state.Alpha.MOs == []:
        make_MOs(molecule, state)
        if settings.SCF.Guess_Mix != 0:
            mix_orbitals(state.Alpha.MOs, state.AlphaOccupancy, settings.SCF.Guess_Mix)
        reference_orbitals = None

    #if settings.SCF.spin_flip:
    #    do_spin_flip(molecule,state)

    make_density_matrices(molecule, state)

#----------------------------------------------------------------------

def make_density_matrices(molecule, this):
    nA = molecule.NAlphaElectrons; nB = molecule.NBetaElectrons
    this.Alpha.Density = this.Alpha.MOs[:,:nA].dot(this.Alpha.MOs[:,:nA].T)
    this.Beta.Density = this.Beta.MOs[:,:nB].dot(this.Beta.MOs[:,:nB].T)
    this.Total.Density = numpy.add(this.Alpha.Density, this.Beta.Density)

#----------------------------------------------------------------------

def calculate_energy(molecule,this):

    energy = 0.0e0

    for mu in range(0,molecule.NOrbitals):
       for nu in range(0,molecule.NOrbitals):
          energy += 0.5e0*(this.Total.Density[mu][nu]*molecule.Core[mu][nu]+
                           this.Alpha.Density[mu][nu]*this.Alpha.Fock[mu][nu]+
                           this.Beta.Density[mu][nu] *this.Beta.Fock[mu][nu])

    this.Energy = energy

#----------------------------------------------------------------------

def make_core_matrices(molecule):

    for shell_pair in molecule.ShellPairs:
        # generate one-electron integrals
        core,overlap = integrals.one_electron(molecule,shell_pair)
        na = shell_pair.Centre1.Cgtf.NAngMom
        nb = shell_pair.Centre2.Cgtf.NAngMom
        ia_vec = shell_pair.Centre1.Ivec
        ib_vec = shell_pair.Centre2.Ivec
        # and pack them away appropriately
        for i in range(0,na):
            for j in range(0,nb):
                molecule.Core[ia_vec[i]][ib_vec[j]] = core[i][j]
                molecule.Overlap[ia_vec[i]][ib_vec[j]] = overlap[i][j]

    # construct and store canonical orthogonalization matrices
    s,U = numpy.linalg.eigh(molecule.Overlap)
    sp = [element**-0.5e0 for element in s]
    molecule.X = numpy.dot(U,numpy.identity(len(sp))*(sp))
    molecule.Xt = numpy.transpose(molecule.X)
    # construct and store half-overlap matrix
    molecule.S = numpy.real(sqrtm(molecule.Overlap))   # sqrt(overlap) sometimes gives very small imaginary components

#----------------------------------------------------------------------

def make_coulomb_exchange_matrices(molecule, this, ints_handling, num_iterations):

    this.Total.Coulomb.fill(0)
    this.Alpha.Exchange.fill(0)
    this.Beta.Exchange.fill(0)

    for shell_pair1 in molecule.ShellPairs:
        ia_vec = shell_pair1.Centre1.Ivec
        ib_vec = shell_pair1.Centre2.Ivec
        for shell_pair2 in molecule.ShellPairs:
            ic_vec = shell_pair2.Centre1.Ivec
            id_vec = shell_pair2.Centre2.Ivec

            # Calculate integrals if direct HF or first iteration
            if (ints_handling == 'DIRECT') or (num_iterations == 1):
                coulomb,exchange = integrals.two_electron(shell_pair1,shell_pair2)

            for m in range(0,shell_pair1.Centre1.Cgtf.NAngMom):
                for n in range(0,shell_pair1.Centre2.Cgtf.NAngMom):
                    for l in range(0,shell_pair2.Centre1.Cgtf.NAngMom):
                        for s in range(0,shell_pair2.Centre2.Cgtf.NAngMom):

                            # Save the integrals on the first pass of an indirect HF job
                            if (ints_handling == 'INCORE') and (num_iterations == 1):
                                molecule.CoulombIntegrals[ia_vec[m]][ib_vec[n]][ic_vec[l]][id_vec[s]] = coulomb[m][n][l][s]
                                molecule.ExchangeIntegrals[ia_vec[m]][id_vec[s]][ic_vec[l]][ib_vec[n]] = exchange[m][s][l][n]
                            ## FUTURE ##
                            # elif (ints_handling == 'ONDISK') and (num_iterations == 1):
                            #   -> dump non-negligible values to file, along with their ia_vec etc indices

                            # Construct coulomb and exchange matrices
                            if (ints_handling == 'INCORE'):
                                this.Total.Coulomb[ia_vec[m]][ib_vec[n]]  +=  this.Total.Density[ic_vec[l]][id_vec[s]]* \
                                                                              molecule.CoulombIntegrals[ia_vec[m],ib_vec[n],ic_vec[l],id_vec[s]]
                                this.Alpha.Exchange[ia_vec[m]][ib_vec[n]] += -this.Alpha.Density[ic_vec[l]][id_vec[s]]* \
                                                                              molecule.ExchangeIntegrals[ia_vec[m],id_vec[s],ic_vec[l],ib_vec[n]]
                                this.Beta.Exchange[ia_vec[m]][ib_vec[n]]  += -this.Beta.Density[ic_vec[l]][id_vec[s]]* \
                                                                              molecule.ExchangeIntegrals[ia_vec[m],id_vec[s],ic_vec[l],ib_vec[n]]
                            ## FUTURE ##
                            # elif (ints_handling == 'ONDISK'):
                            #   ->  reload orbitals from file
                            else:
                                this.Total.Coulomb[ia_vec[m]][ib_vec[n]]  +=  this.Total.Density[ic_vec[l]][id_vec[s]]*coulomb[m][n][l][s]
                                this.Alpha.Exchange[ia_vec[m]][ib_vec[n]] += -this.Alpha.Density[ic_vec[l]][id_vec[s]]*exchange[m][s][l][n]
                                this.Beta.Exchange[ia_vec[m]][ib_vec[n]]  += -this.Beta.Density[ic_vec[l]][id_vec[s]]*exchange[m][s][l][n]

        for m in range(0,shell_pair1.Centre1.Cgtf.NAngMom):
            for n in range(0,shell_pair1.Centre2.Cgtf.NAngMom):
                this.Alpha.Fock[ia_vec[m]][ib_vec[n]] += (this.Total.Coulomb[ia_vec[m]][ib_vec[n]] + this.Alpha.Exchange[ia_vec[m]][ib_vec[n]])
                this.Beta.Fock[ia_vec[m]][ib_vec[n]] += (this.Total.Coulomb[ia_vec[m]][ib_vec[n]] + this.Beta.Exchange[ia_vec[m]][ib_vec[n]])

#----------------------------------------------------------------------

def constrain_UHF(molecule, this, state_index):

    N = molecule.NElectrons
    Nab = molecule.NAlphaElectrons * molecule.NBetaElectrons
    S = molecule.S                                              # Note: S is the sqrt of the atomic overlap matrix

    half_density_matrix = S.dot(this.Total.Density/2).dot(S)
    NO_vals, NO_vects = numpy.linalg.eigh(half_density_matrix)  # See J. Chem. Phys. 88, 4926
    NO_coeffs = numpy.linalg.inv(S).dot(NO_vects)               # for details on finding the NO coefficents
    back_trans = numpy.linalg.inv(NO_coeffs)

    # Selecting the various spaces based on natural orbital occupancy
    #core_space = [i for i, occ in enumerate(NO_vals) if occ >= (1 - c.CUHF_thresh)]
    #valence_space = [i for i, occ in enumerate(NO_vals) if occ <= c.CUHF_thresh]

    # Selecting the spaces based on occupany vectors
    total_occ = [x[0] + x[1] for x in zip(this.AlphaOccupancy, this.BetaOccupancy)]
    core_space = range(molecule.NOrbitals)[-total_occ.count(2):]
    valence_space = range(molecule.NOrbitals)[:total_occ.count(0)]

    delta = (this.Alpha.Fock - this.Beta.Fock) / 2
    delta = NO_coeffs.T.dot(delta).dot(NO_coeffs)    # Transforming delta into the NO basis
    lambda_matrix = numpy.zeros(numpy.shape(delta))
    for i in core_space:
        for j in valence_space:
            lambda_matrix[i,j] = -delta[i,j]
            lambda_matrix[j,i] = -delta[j,i]
    lambda_matrix = back_trans.T.dot(lambda_matrix).dot(back_trans)  # Transforming lambda back to the AO basis

    # Need to zero out the small elements of the lambda matrix otherwise lose of significant
    # figures in the small elements of the Beta Fock matrix can cause difference in calculations
    # with spin symmetric excitations
    small_lambda_values = abs(lambda_matrix) < c.lambda_zero_thresh
    lambda_matrix[small_lambda_values] = 0.0

    this.Alpha.Fock = this.Alpha.Fock + lambda_matrix
    this.Beta.Fock = this.Beta.Fock - lambda_matrix

def calc_spin(molecule, state):
    Na = molecule.NAlphaElectrons; Nb = molecule.NBetaElectrons
    S_RHF = 0.25 * (Na - Nb) * (Na - Nb + 2)
    overlap = state.Alpha.MOs[:,:Na].T.dot(molecule.Overlap).dot(state.Beta.MOs [:,:Nb])
    contamination = Nb - numpy.square(overlap).sum()
    S_UHF = S_RHF + contamination
    state.S2 = S_UHF

#----------------------------------------------------------------------

def mix_orbitals(MOs, occupancy, mix_coeff):
    """ Mixes the HOMO and LUMO of the alpha orbitals by an ammount specified
    by mix, 0 is not mixing, 1 is replacing the HOMO with the LUMO """
    HOMO = numpy.count_nonzero(occupancy) - 1
    MOs[:,HOMO] = MOs[:,HOMO] * (1 - mix_coeff) + MOs[:,HOMO+1] * mix_coeff
#-----------------------------------------------------------------

def plot(energies, start=1):
    import matplotlib.pyplot as plt
    x = range(start,len(energies)+1)
    energies = energies[start-1:]
    #energies = [e * 2625.5 for e in energies] # Convert to kJ/mol
    plt.plot(x, energies, marker='o')
    #plt.ylim((-3600, -3900))
    plt.ylabel("Energies (Hartrees)")
    plt.xlabel("SCF Iterations")
    plt.show()

def make_density_matrices(molecule, this):
    nA = molecule.NAlphaElectrons; nB = molecule.NBetaElectrons
    this.Alpha.Density = this.Alpha.MOs[:,:nA].dot(this.Alpha.MOs[:,:nA].T)
    this.Beta.Density = this.Beta.MOs[:,:nB].dot(this.Beta.MOs[:,:nB].T)
    this.Total.Density = numpy.add(this.Alpha.Density, this.Beta.Density)

def eigenspace_update(spin):
    """ Updating one set of MO using Guided Hartree Fock
        see J. Chem. Theory Comput. 9, 3933 """
    delta = spin.MOs.T.dot(spin.Fock).dot(spin.MOs)
    spin.Energies, U = numpy.linalg.eigh(delta)
    spin.MOs = spin.MOs.dot(U)

# Possibly break this into two functions, 1 that looks for two matching states
# and returns their indices if found and another that does the actual swap
def check_swapped_spin(molecule, state, state_index, settings):
    """ Looks a state that resemble the target state with spin labels swapped and
        - if found constructs the state and prints the HF output before returning
        true, if no matching state is found it returns false """
    for (i, old_state) in enumerate(molecule.States[:state_index]):
        alpha_match = old_state.AlphaOccupancy == state.BetaOccupancy
        beta_match = old_state.BetaOccupancy == state.AlphaOccupancy
        if alpha_match and beta_match:
            state = copy.deepcopy(old_state)
            state.Alpha, state.Beta = state.Beta, state.Alpha
            print_message = "Constructed from state {}".format(i)
            printf.HF_Loop(state, settings, print_message, print_message, True)
            printf.HF_Final(settings)
            return True, state
    return False, state

def do_spin_flip(molecule, state):
    """ Permuting the occupancy vectors and the the number of electrons of each
    spin in the copied state instance (not in molecule.States[state_index])"""
    # Number of alpha electrons in the spin flipped configuration
    n_alpha = [i for (i,occ) in enumerate(state.AlphaOccupancy) if occ][-1] + 1
    new_alpha = [1] * n_alpha + [0] * (len(state.AlphaOccupancy) - n_alpha)
    n_alpha = sum(new_alpha)
    n_beta = molecule.NElectrons - n_alpha
    new_beta = [1] * n_beta + [0] * (len(state.BetaOccupancy) - n_beta)

    state.AlphaOccupancy, state.BetaOccupancy = new_alpha, new_beta
    molecule.NAlphaElectrons, molecule.NBetaElectrons = n_alpha, n_beta

def undo_spin_flip(molecule, state, state_index):

    # Ensure the alpha and beta orbitals are in phase
    MO1_overlap = state.Alpha.MOs[:,0].dot(molecule.Overlap).dot(state.Beta.MOs[:,0])
    if MO1_overlap < 0:
        state.Beta.MOs *= -1

    # Sort all the beta orbitals by energy then replace every unoccupied (unoptimized)
    # with it's alpha counterpart if that is occupied (optimized)
    util.sort_MOs(state.Alpha, molecule)
    for i in range(molecule.NOrbitals):
        if state.AlphaOccupancy[i] and not state.BetaOccupancy[i]:
            state.Beta.MOs[:,i] = state.Alpha.MOs[:,i]
            state.Beta.Energies[i] = state.Alpha.Energies[i]
        elif state.AlphaOccupancy[i] and state.BetaOccupancy[i]:
            average_MO = (state.Alpha.MOs[:,i] + state.Beta.MOs[:,i]) / 2
            state.Alpha.MOs[:,i] = average_MO
            state.Beta.MOs[:,i] = average_MO

    # Reload the occupancy information from the unflipped state instance
    # stored in 'molecule'
    state.AlphaOccupancy = molecule.States[state_index].AlphaOccupancy
    state.BetaOccupancy = molecule.States[state_index].BetaOccupancy
    molecule.NAlphaElectrons = sum(state.AlphaOccupancy)
    molecule.NBetaElectrons = sum(state.BetaOccupancy)

    # Ensure that the orbitals are ordered correctly
    # Finding the arrangement of indexes to ensure the occupied orbitals are
    # in the first coloums of the array
    alpha_idx = (-1 * numpy.array(state.AlphaOccupancy)).argsort(kind="mergesort")    # Using mergesort for sorting stability
    state.Alpha.MOs = state.Alpha.MOs[:,alpha_idx]
    state.Alpha.Energies = state.Alpha.Energies[alpha_idx]
    beta_idx = (-1 * numpy.array(state.BetaOccupancy)).argsort(kind="mergesort")      # Using mergesort for sorting stability
    state.Beta.MOs = state.Beta.MOs[:,beta_idx]
    state.Beta.Energies = state.Beta.Energies[beta_idx]

    make_density_matrices(molecule, state)
    calculate_energy(molecule, state)
    state.TotalEnergy = state.Energy + molecule.NuclearRepulsion
    calc_spin(molecule,state)
    return state

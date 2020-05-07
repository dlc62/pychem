# System libraries
import numpy
# TODO remove this 
import numpy as np 

# Custom-written code
from Methods import noci
from Util.structures import Spin 

#=================================================================#
#                                                                 #
#                        Main Function                            #
#                                                                 #
#=================================================================#

def do(alpha_MOs, beta_MOs, alpha_core, beta_core, settings, molecule):

    Ca1 = alpha_MOs[0]
    Ca2 = alpha_MOs[1]
    Cb1 = beta_MOs[0]
    Cb2 = beta_MOs[1]
     
    # There are two approches to this, diagonalizing the core hamiltonian matrix 
    # (Won't this be the same for each state) and actually forming a fock matrix 
    # from the MOs, but it's not clear what to do with the two different sets of MOs 
    # in that case, maybe use the co-density matrix?
    Ea = get_energies(alpha_core, molecule)
    Eb = get_energies(beta_core, molecule)

    # Find the required overlap terms 
    alpha_overlaps = numpy.diagonal(Ca1.T.dot(molecule.Overlap).dot(Ca2))
    beta_overlaps = numpy.diagonal(Ca1.T.dot(molecule.Overlap).dot(Cb2))

    reduced_overlap, zeros_list = noci.process_overlaps(1, [], alpha_overlaps[:molecule.NAlphaElectrons], Spin.Alpha)
    reduced_overlap, zeros_list = noci.process_overlaps(reduced_overlap, zeros_list[:molecule.NBetaElectrons], beta_overlaps, Spin.Beta)
    num_zeros = len(zeros_list)

    if num_zeros > 2:
        return 0

    # Perform transformation of coulomb and exchange matrices from AO to MO basis
    # Allocate space to store half-transformed and fully-transformed orbitals (in MO basis)

    Alpha = numpy.zeros((molecule.NOrbitals,) * 4)
    Beta = numpy.zeros((molecule.NOrbitals,) * 4)
    AlphaBeta = numpy.zeros((molecule.NOrbitals,) * 4)

    # Do first half-transformation

    for m in range(0, molecule.NOrbitals):
      for l in range(0, molecule.NOrbitals):

        X = molecule.CoulombIntegrals[m,:,l,:]

        for b in range(0, molecule.NOrbitals):
          for d in range(0, molecule.NOrbitals):
            Alpha[m,b,l,d]     = (Ca1[:,b].T).dot(X.dot(Ca2[:,d]))  
            Beta[m,b,l,d]      = (Cb1[:,b].T).dot(X.dot(Cb2[:,d]))  

            AlphaBeta[m,b,l,d] = (Ca1[:,b].T).dot(X.dot(Cb1[:,d]))  
 
    # Complete transformation replacing m,b,l,d elements with a,b,c,d

    for b in range(0, molecule.NOrbitals):
      for d in range(0, molecule.NOrbitals):
        
        # TODO why are these being copied?
        Yaa = Alpha[:,b,:,d].copy()
        Yab = AlphaBeta[:,b,:,d].copy()
        Ybb = Beta[:,b,:,d].copy()

        for a in range(0, molecule.NOrbitals):
          for c in range(0, molecule.NOrbitals):

            Alpha[a,b,c,d]     = (Ca1[:,a].T).dot(Yaa.dot(Ca2[:,c])) 
            Beta[a,b,c,d]      = (Cb1[:,a].T).dot(Ybb.dot(Cb2[:,c])) 

            AlphaBeta[a,b,c,d] = (Ca2[:,a].T).dot(Yab.dot(Cb2[:,c])) 

    # Use the transformed integrals in the MP2 energy expression, i,j index occupied MOs and p,q are virtuals
    if num_zeros == 0:
        mp2_energy = no_zeros(Alpha, Beta, AlphaBeta, Ea, Eb, molecule, alpha_overlaps, beta_overlaps, reduced_overlap)
    elif num_zeros == 1:
        mp2_energy = one_zero(Alpha, Beta, AlphaBeta, Ea, Eb, molecule, alpha_overlaps, beta_overlaps, reduced_overlap, zeros_list[0])
    elif num_zeros == 2:
        mp2_energy = two_zeros(Alpha, Beta, AlphaBeta, Ea, Eb, molecule, alpha_overlaps, beta_overlaps, reduced_overlap, zeros_list)
    else:
        mp2_energy = 0
    print(mp2_energy)

    return mp2_energy

def no_zeros(Alpha, Beta, AlphaBeta, Ea, Eb, molecule, alpha_overlaps, beta_overlaps, reduced_overlap):

    MP2_Eaa = eval_integrals_no_zero(molecule.NAlphaElectrons, molecule.NOrbitals, Ea, 
                                     reduced_overlap, alpha_overlaps, Alpha)

    MP2_Ebb = eval_integrals_no_zero(molecule.NBetaElectrons, molecule.NOrbitals, Eb, 
                                     reduced_overlap, beta_overlaps, Beta)

    MP2_Eab = 0 
    for i in range(molecule.NAlphaElectrons):
      for j in range(molecule.NBetaElectrons):
        for p in range(molecule.NAlphaElectrons, molecule.NOrbitals):
          for q in range(molecule.NBetaElectrons, molecule.NOrbitals):
            this_overlap = reduced_overlap / (alpha_overlaps[i] * beta_overlaps[j])
            MP2_Eab += (AlphaBeta[i,p,j,q])**2 / (Ea[i] + Eb[j] - Ea[p] - Eb[q]) * this_overlap 
    
    return MP2_Eaa + MP2_Eab + MP2_Ebb 

def one_zero(Alpha, Beta, AlphaBeta, Ea, Eb, molecule, alpha_overlaps, beta_overlaps, reduced_overlap, zero):
    zero_index, zero_spin = zero

    if zero_spin == Spin.Alpha:
        MP2_Eaa = eval_integrals_one_zero(molecule.NAlphaElectrons, molecule.NOrbitals, Ea,
                                         reduced_overlap, alpha_overlaps, Alpha, zero_index)

        MP2_Ebb = eval_integrals_no_zero(molecule.NBetaElectrons, molecule.NOrbitals, Eb, 
                                         reduced_overlap, beta_overlaps, Beta)

        MP2_Eab = eval_AlphaBeta_one_zero(molecule.NBetaElectrons, molecule.NOrbitals, Eb,
                                          reduced_overlap, beta_overlaps, AlphaBeta, zero_index)

    elif zero_spin == Spin.Beta:
        MP2_Eaa = eval_integrals_no_zero(molecule.NAlphaElectrons, molecule.NOrbitals, Ea, 
                                         reduced_overlap, alpha_overlaps, Alpha)

        MP2_Ebb = eval_integrals_one_zero(molecule.NBetaElectrons, molecule.NOrbitals, Ea,
                                         reduced_overlap, beta_overlaps, Beta, zero_index)

        MP2_Eab = eval_AlphaBeta_one_zero(molecule.NAlphaElectrons, molecule.NOrbitals, Ea,
                                          reduced_overlap, alpha_overlaps, AlphaBeta, zero_index)

    return MP2_Eaa + MP2_Eab + MP2_Ebb 

def two_zeros(Alpha, Beta, AlphaBeta, Ea, Eb, molecule, alpha_overlaps, beta_overlaps, reduced_overlap, zeros_list):

    # Will the ground states contributes in this case?
    # Two cases 
    # 1: Both of the zeros are of the same spin 
    #    Only those indices contribute in that spin 
    #    The AlphaBeta terms are zero
    # 2: The two zeros are of different spin
    #    Both single spin terms are as in one_zero 
    #    The AlphaBeta terms are the same as the one spin terms in the first case   

    # case 1
    if zeros_list[0].spin == zeros_list[1].spin:

        if zeros_list[0].spin == Spin.Alpha:
            MP2_Eaa = eval_integrals_two_zeros(molecule.Alpha.NAlphaElectrons, molecule.NOrbitals, Ea,
                                               reduced_overlap, alpha_overlaps, Alpha, zeros_list)

            MP2_Ebb = eval_integrals_no_zero(molecule.NBetaElectrons, molecule.NOrbitals, Eb, 
                                             reduced_overlap, beta_overlaps, Beta)

            MP2_Eab = 0 

        if zeros_list[0].spin == Spin.Beta:
            MP2_Ebb = eval_integrals_two_zeros(molecule.NBetaElectrons, molecule.NOrbitals, Eb,
                                               reduced_overlap, beta_overlaps, Beta, zeros_list)

            MP2_Eaa = eval_integrals_no_zero(molecule.NAlphaElectrons, molecule.NOrbitals, Ea, 
                                             reduced_overlap, alpha_overlaps, Alpha)

            MP2_Eab = 0

    # case 2 
    else: 
        MP2_Eaa = 0 
        MP2_Ebb = 0

        i = zeros_list[0].index
        j = zeros_list[1].index
        MP2_Eab = AlphaBeta[i,i,j,j]**2 * reduced_overlap 

    return MP2_Eaa + MP2_Ebb + MP2_Eab 


# TODO Sort out which arguments are needed here 
def eval_integrals_two_zeros(NElectrons, NOrbitals, E, reduced_overlap, overlaps, integrals, zeros_list):
    i = zeros_list[0].index
    j = zeros_list[1].index
    term = (integrals[i,i,j,j] - integrals[i,j,j,i])**2 * reduced_overlap
    return term

def eval_integrals_one_zero(NElectrons, NOrbitals, E, reduced_overlap, overlaps, integrals, zero_index):
    term = 0 
    z = zero_index
    for i in range(0, NElectrons):
        for p in range(NElectrons, NOrbitals):
            this_overlap = reduced_overlap / overlaps[i] 
            term += (integrals[i,p,z,z] - integrals[i,z,z,p])**2 / (E[i] - E[p]) * this_overlap 
    return term

def eval_integrals_no_zero(NElectrons, NOrbitals, E, reduced_overlap, overlaps, integrals): 
    term = 0
    for i in range(0, NElectrons):
      for j in range(0, i+1):
        for p in range(NElectrons, NOrbitals):
          for q in range(NElectrons, p+1):
            this_overlap = reduced_overlap / (overlaps[i] * overlaps[j])
            term += (integrals[i,p,j,q] - integrals[i,q,j,p])**2 / (E[i] + E[j] - E[p] - E[q]) * this_overlap
    return term 

def eval_AlphaBeta_one_zero(NElectrons, NOrbitals, E, reduced_overlap, overlaps, AlphaBeta, zero_index):
    z = zero_index 
    term = 0
    for i in range(NElectrons):
      for p in range(NElectrons, NOrbitals):
        this_overlap = reduced_overlap / overlaps[i]
        term += AlphaBeta[i, p, z, z]**2 / (E[i] - E[p]) * this_overlap
    return term

def get_energies(core_matrix, molecule):
    X = molecule.X
    Xt = molecule.Xt
    energies, _ = numpy.linalg.eigh(Xt.dot(core_matrix).dot(X))
    return energies

def get_energies2(alpha_MOs, beta_MOs, nE, molecule):

    nA = molecule.NAlphaElectrons; nB = molecule.NBetaElectrons
    alpha_density = alpha_MOs[:,:nA].dot(alpha_MOs[:,:nA].T)
    beta_density  = beta_MOs[:,:nB].dot(beta_MOs[:,:nB].T)
    total_density = alpha_density + beta_density 

    coulomb  = numpy.einsum("cd,abcd -> ab", total_density, molecule.CoulombIntegrals)
    alpha_exchange = numpy.einsum("cb,abcd -> ad", -alpha_density, molecule.CoulombIntegrals)
    beta_exchange  = numpy.einsum("cb,abcd -> ad", -beta_density, molecule.CoulombIntegrals)

    alpha_fock = molecule.Core + coulomb + alpha_exchange
    beta_fock = molecule.Core + coulomb + beta_exchange

    X = molecule.X
    Xt = molecule.Xt
    alpha_energies, _ = numpy.linalg.eigh(Xt.dot(alpha_fock).dot(X))
    beta_energies, _ = numpy.linalg.eigh(Xt.dot(beta_fock).dot(X))

    return alpha_energies, beta_energies

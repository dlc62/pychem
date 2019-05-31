# System libraries
from __future__ import print_function
import numpy
import sys
from itertools import izip

# Custom-written data modules
from Data import constants

# Custom-written utility modules
from Util import printf

# Custom-written code modules
import integrals
import hf_extensions as hf

#=================================================================#
#                                                                 #
#                        Main Function                            #
#                                                                 #
#=================================================================#

def do(settings, molecule, basis_set, state_index, initial_run = True):
    # Calculate values that are constant throughout the calculation
    molecule.NuclearRepulsion = integrals.nuclear_repulsion(molecule)
    make_core_matrices(molecule)

    # Set up for SCF calculation
    this_state = molecule.States[state_index]
    initialize_fock_matrices(molecule.Core, this_state)
    if initial_run:
       evaluate_2e_ints(molecule)
   
    # Generate initial orbitals and/or density matrices
    if settings.SCF.Guess == "READ" and basis_set == settings.BasisSets[0]:
        try:
            # TODO make this more flexible, maybe support only providing MO files for some states 
            # and fall back to CORE for the rest
            this_state.Alpha.MOs = numpy.loadtxt(settings.SCF.AlphaMOFile[state_index])
            this_state.Beta.MOs = numpy.loadtxt(settings.SCF.BetaMOFile[state_index])
            assert len(this_state.Alpha.MOs) == molecule.NOrbitals
            assert len(this_state.Beta.MOs) == molecule.NOrbitals
            reference_orbitals = [this_state.Alpha.MOs, this_state.Beta.MOs]
        except IndexError:
            print("No MO file given for state {}".format(state_index))
            sys.exit()
        except IOError: 
            print("Could not read MO file, check you have the name right")
            sys.exit()
        except AssertionError:
            print('Error: Incorrect MOs supplied - check basis set and supply both alpha and beta sets')
            sys.exit()
    elif this_state.Alpha.MOs != []:
        reference_orbitals = [this_state.Alpha.MOs, this_state.Beta.MOs]
    elif settings.SCF.Guess == "CORE":
        make_MOs(molecule, this_state)
        reference_orbitals = None

    make_density_matrices(molecule, this_state)
    
    # Calculate initial energy
    calculate_energy(molecule, this_state)
    dE = 1                                    # an arbitaray number larger than the convergence threshold

    # Initial print
    if settings.PrintLevel == "VERBOSE":
       printf.delimited_text(settings.OutFile," Hartree-Fock initialization ")
       printf.text_value(settings.OutFile, " Nuclear repulsion energy: ", molecule.NuclearRepulsion)
       printf.text_value(settings.OutFile, " Alpha density matrix ", this_state.Alpha.Density,
                                           " Beta density matrix ", this_state.Beta.Density)
       printf.text_value(settings.OutFile, " Alpha MO coefficients ", this_state.Alpha.MOs,
                                           " Beta MO coefficients ", this_state.Beta.MOs)
    printf.delimited_text(settings.OutFile," Hartree-Fock iterations ") 

    #-------------------------------------------#
    #           Begin SCF Iterations            #
    #-------------------------------------------#
    num_iterations = 0
    diis_error = None

    while constants.energy_convergence < abs(dE):
        num_iterations += 1

        # Don't print non converging message on zero iteration runs 
        # For using orbitals from other souces without doing SCF 
        if num_iterations > settings.SCF.MaxIter:
            if settings.SCF.MaxIter > 0:
                print("SCF not converging")
            break

        #-------------------------------------------#
        #               Main SCF step               #
        #-------------------------------------------#
#        initialize_fock_matrices(molecule.Core, this_state)
        make_coulomb_exchange_matrices(molecule, this_state)
        make_fock_matrices(molecule, this_state)
         
        # apply CUHF constraints
        if settings.SCF.Reference == "CUHF" or (settings.SCF.ConstrainExcited and state_index != 0):
            constrain_UHF(molecule, this_state)

        #-------------------------------------------#
        #    Convergence accelerators/modifiers     #
        #-------------------------------------------#
        # DIIS
        if settings.DIIS.Use and num_iterations > settings.DIIS.Start:
            hf.diis.do(molecule, this_state, settings)
            diis_error = max(this_state.AlphaDIIS.Error, this_state.BetaDIIS.Error)

        # Update MOM reference orbitals to last iteration values if requested
        if settings.MOM.Use:
            if settings.MOM.Reference == 'MUTABLE':
                reference_orbitals = [this_state.Alpha.MOs, this_state.Beta.MOs]
        
        make_MOs(molecule, this_state)

        # Optionally, use MOM to reorder MOs but absolutely not if doing NOCI calculations
        if settings.MOM.Use and reference_orbitals != None:# and not settings.Method.startswith("NOCI"):
            hf.mom.do(molecule, this_state, reference_orbitals)

        # Sort occupied orbitals by energy
        # This isn't nessecary for HF but avoids compilcations in 
        # post-HF methods and makes the output easier to follow
        this_state.Alpha.sort_occupied()
        this_state.Beta.sort_occupied()

       #-------------------------------------------#

        make_density_matrices(molecule,this_state)

        old_energy = this_state.Energy
        calculate_energy(molecule, this_state)
        dE = this_state.Energy - old_energy
        this_state.TotalEnergy = this_state.Energy + molecule.NuclearRepulsion

        # Loop print
        printf.text_value(settings.OutFile, " Cycle: ", num_iterations, " Total energy: ", this_state.TotalEnergy,
                          " Change in energy: ", dE, " DIIS error: ", diis_error)
        if settings.PrintLevel == "VERBOSE":
           print_intermediates(settings.OutFile, this_state, (settings.SCF.Reference == "RHF"))



    # Final print/dump to file
    printf.delimited_text(settings.OutFile, " End of Hartree-Fock iterations ")
    if this_state.S2 is not None:
        printf.text(settings.OutFile, "<S^2> = %.2f" % this_state.S2)
    printf.text_value(settings.OutFile, 'Final HF energy: ', this_state.TotalEnergy)
    printf.blank_line(settings.OutFile)
    print_intermediates(settings.OutFile, this_state, (settings.SCF.Reference == "RHF"))
    printf.delimited_text(settings.OutFile, " End of Hartree-Fock calculation ")
    if settings.DumpMOs:
       numpy.savetxt(basis_set+'_'+str(state_index)+'.alpha_MOs',this_state.Alpha.MOs)
       numpy.savetxt(basis_set+'_'+str(state_index)+'.beta_MOs',this_state.Beta.MOs)

#############################################################################
########################### Basic HF Subroutines ############################
# this = this electronic state


def initialize_fock_matrices(core,this):

    this.Alpha.Fock = core[:,:]
    this.Beta.Fock = core[:,:]

#----------------------------------------------------------------------

def make_MOs(molecule,this):

    X = molecule.X
    Xt = molecule.Xt
    this.Alpha.Energies,Ca = numpy.linalg.eigh(Xt.dot(this.Alpha.Fock).dot(X))
    this.Beta.Energies,Cb = numpy.linalg.eigh(Xt.dot(this.Beta.Fock).dot(X))
    this.Alpha.MOs = numpy.dot(X,Ca)
    this.Beta.MOs = numpy.dot(X,Cb)

#----------------------------------------------------------------------

def make_density_matrices(molecule,this):

    nA = this.NAlpha; nB = this.NBeta
    this.Alpha.Density = this.Alpha.MOs[:,:nA].dot(this.Alpha.MOs[:,:nA].T)
    this.Beta.Density  = this.Beta.MOs[:,:nB].dot(this.Beta.MOs[:,:nB].T)
    this.Total.Density = this.Alpha.Density + this.Beta.Density
   
#----------------------------------------------------------------------

def calculate_energy(molecule,this):

    #energy = 0.0e0

    #for a in range(0,molecule.NOrbitals):
    #   for b in range(0,molecule.NOrbitals):
    #      energy += 0.5e0*(this.Total.Density[a,b]*molecule.Core[a,b]+
    #                       this.Alpha.Density[a,b]*this.Alpha.Fock[a,b]+
    #                       this.Beta.Density[a,b] *this.Beta.Fock[a,b])
    core = numpy.sum(numpy.multiply(this.Total.Density, molecule.Core))
    alpha = numpy.sum(numpy.multiply(this.Alpha.Density, this.Alpha.Fock))
    beta = numpy.sum(numpy.multiply(this.Beta.Density, this.Beta.Fock))

    this.Energy = 0.5 * (core + alpha + beta)
    
    #this.Energy = energy

#----------------------------------------------------------------------

def make_core_matrices(molecule):

    for a in range(0,molecule.NCgtf):
      for b in range(a,molecule.NCgtf):

        shell_pair = molecule.ShellPairs[(a,b)]
        ia_vec = shell_pair.Centre1.Ivec
        ib_vec = shell_pair.Centre2.Ivec

        core,overlap = integrals.one_electron(molecule,shell_pair)

        for i in range(0,len(ia_vec)):
            for j in range(0,len(ib_vec)):
                molecule.Core[ia_vec[i]][ib_vec[j]] = core[i][j]
                molecule.Core[ib_vec[j]][ia_vec[i]] = core[i][j]
                molecule.Overlap[ia_vec[i]][ib_vec[j]] = overlap[i][j]
                molecule.Overlap[ib_vec[j]][ia_vec[i]] = overlap[i][j]

    # construct and store canonical orthogonalization matrices
    s,U = numpy.linalg.eigh(molecule.Overlap)
    U = U[:,s.argsort()[::-1]]
    s = numpy.sort(s)[::-1]
    sp = [element**-0.5e0 for element in s if element > constants.linear_dependence]
    nsp = len(sp)
    molecule.X = numpy.dot(U[:,:nsp],numpy.identity(nsp)*sp)
    molecule.Xt = numpy.transpose(molecule.X)
    # construct and store half-overlap matrix
    sr = numpy.zeros((nsp,nsp))
    for i,element in enumerate(sp):
       sr[i,i] = 1/element 
    molecule.S = (U[:,:nsp]).dot(sr).dot(U[:,:nsp].T)

#----------------------------------------------------------------------

def evaluate_2e_ints(molecule,ints_type=0,grid_value=-1.0):

    # Compute Schwarz bounds and store integrals generated in process
    for a in range(0,molecule.NCgtf):
      for b in range(a,molecule.NCgtf):

        ab = molecule.ShellPairs[(a,b)]; 
        ia_vec = ab.Centre1.Ivec; ib_vec = ab.Centre2.Ivec
        bounds = numpy.zeros((len(ia_vec),len(ib_vec)))
        coulomb = integrals.two_electron(ab,ab,ints_type,grid_value)
        for m in range(0,len(ia_vec)):
          for n in range(0,len(ib_vec)):
            bounds[(m,n)] = coulomb[m][n][m][n]
        molecule.Bounds[a][b] = numpy.sqrt(bounds)
        
        #for m in range(0,len(ia_vec)):
        #  for n in range(0,len(ib_vec)):
        #    for l in range(0,len(ia_vec)):
        #      for s in range(0,len(ib_vec)):
        #        molecule.CoulombIntegrals[ (ia_vec[m], ib_vec[n], ia_vec[l], ib_vec[s]) ] = coulomb[m][n][l][s]
        #        molecule.CoulombIntegrals[ (ib_vec[n], ia_vec[m], ia_vec[l], ib_vec[s]) ] = coulomb[m][n][l][s]
        #        molecule.CoulombIntegrals[ (ia_vec[m], ib_vec[n], ib_vec[s], ia_vec[l]) ] = coulomb[m][n][l][s]
        #        molecule.CoulombIntegrals[ (ib_vec[n], ia_vec[m], ib_vec[s], ia_vec[l]) ] = coulomb[m][n][l][s]

        # This does the same as above but is substantially faster
        for m, coul1 in izip(ia_vec, coulomb):
          for n, coul2 in izip(ib_vec, coul1):
            for l, coul3 in izip(ia_vec, coul2):
              for s, coul_val in izip(ib_vec, coul3):
                molecule.CoulombIntegrals[m, n, l, s] = coul_val
                molecule.CoulombIntegrals[n, m, l, s] = coul_val
                molecule.CoulombIntegrals[m, n, s, l] = coul_val
                molecule.CoulombIntegrals[n, m, s, l] = coul_val

    # Evaluate and store all other non-negligible 2e ints
    for a in range(0,molecule.NCgtf):
      for b in range(a,molecule.NCgtf):

        ab = molecule.ShellPairs[(a,b)]  
        ab_bound = molecule.Bounds[a][b]
        ia_vec = ab.Centre1.Ivec; ib_vec = ab.Centre2.Ivec

        for c in range(a,molecule.NCgtf):
          for d in range(c,molecule.NCgtf):

            cd = molecule.ShellPairs[(c,d)]  
            cd_bound = molecule.Bounds[c][d]
            ic_vec = cd.Centre1.Ivec; id_vec = cd.Centre2.Ivec

            if (a == c) and (b == d):
              coulomb = None      # Already done
            else:
              coulomb_bound = numpy.multiply.outer(ab_bound,cd_bound)
              if numpy.amax(coulomb_bound) > constants.integral_threshold:
                coulomb = integrals.two_electron(ab,cd,ints_type,grid_value)
              else:
                coulomb = None    # Already initialized to zero

            if coulomb is not None:
 
               #for m in range(0,len(ia_vec)):
               # for n in range(0,len(ib_vec)):
               #   for l in range(0,len(ic_vec)):
               #     for s in range(0,len(id_vec)):
               #       molecule.CoulombIntegrals[ (ia_vec[m], ib_vec[n], ic_vec[l], id_vec[s]) ] = coulomb[m][n][l][s]
               #       molecule.CoulombIntegrals[ (ib_vec[n], ia_vec[m], ic_vec[l], id_vec[s]) ] = coulomb[m][n][l][s]
               #       molecule.CoulombIntegrals[ (ia_vec[m], ib_vec[n], id_vec[s], ic_vec[l]) ] = coulomb[m][n][l][s]
               #       molecule.CoulombIntegrals[ (ib_vec[n], ia_vec[m], id_vec[s], ic_vec[l]) ] = coulomb[m][n][l][s]
               #       molecule.CoulombIntegrals[ (ic_vec[l], id_vec[s], ia_vec[m], ib_vec[n]) ] = coulomb[m][n][l][s]
               #       molecule.CoulombIntegrals[ (ic_vec[l], id_vec[s], ib_vec[n], ia_vec[m]) ] = coulomb[m][n][l][s]
               #       molecule.CoulombIntegrals[ (id_vec[s], ic_vec[l], ia_vec[m], ib_vec[n]) ] = coulomb[m][n][l][s]
               #       molecule.CoulombIntegrals[ (id_vec[s], ic_vec[l], ib_vec[n], ia_vec[m]) ] = coulomb[m][n][l][s]

              for m, coul1 in izip(ia_vec,coulomb):
                for n, coul2 in izip(ib_vec, coul1):
                  for l, coul3 in izip(ic_vec, coul2):
                    for s, coul_val in izip(id_vec, coul3):
                      molecule.CoulombIntegrals[m, n, l, s] = coul_val
                      molecule.CoulombIntegrals[n, m, l, s] = coul_val
                      molecule.CoulombIntegrals[m, n, s, l] = coul_val
                      molecule.CoulombIntegrals[n, m, s, l] = coul_val
                      molecule.CoulombIntegrals[l, s, m, n] = coul_val
                      molecule.CoulombIntegrals[l, s, n, m] = coul_val
                      molecule.CoulombIntegrals[s, l, m, n] = coul_val
                      molecule.CoulombIntegrals[s, l, n, m] = coul_val
    
#----------------------------------------------------------------------

def make_coulomb_exchange_matrices(molecule, this):
    
    #this.Total.Coulomb.fill(0)
    #this.Alpha.Exchange.fill(0)
    #this.Beta.Exchange.fill(0)

    #for a in range(0,molecule.NOrbitals):
    #  for b in range(0,molecule.NOrbitals):
    #    for c in range(0,molecule.NOrbitals):
    #      for d in range(0,molecule.NOrbitals):
    #
    #         this.Total.Coulomb[a,b]  +=  this.Total.Density[c,d]*molecule.CoulombIntegrals[a,b,c,d]
    #         this.Alpha.Exchange[a,d] += -this.Alpha.Density[c,b]*molecule.CoulombIntegrals[a,b,c,d]
    #         this.Beta.Exchange[a,d]  += -this.Beta.Density[c,b]*molecule.CoulombIntegrals[a,b,c,d]

    # This does the same as the nexted loops above but 10x faster 
    this.Total.Coulomb  = numpy.einsum("cd,abcd -> ab", this.Total.Density, molecule.CoulombIntegrals)
    this.Alpha.Exchange = numpy.einsum("cb,abcd -> ad", -this.Alpha.Density, molecule.CoulombIntegrals)
    this.Beta.Exchange  = numpy.einsum("cb,abcd -> ad", -this.Beta.Density, molecule.CoulombIntegrals)


#----------------------------------------------------------------------

def make_fock_matrices(molecule, this):

    this.Alpha.Fock = molecule.Core + this.Total.Coulomb + this.Alpha.Exchange
    this.Beta.Fock = molecule.Core + this.Total.Coulomb + this.Beta.Exchange

#----------------------------------------------------------------------

def constrain_UHF(molecule, this):

    occupancy = numpy.add(this.Alpha.Occupancy, this.Beta.Occupancy)
    N = molecule.NElectrons
    Nab = this.NAlpha * this.NBeta
    Na = numpy.count_nonzero(occupancy == 1)                    # Dimension of active space
    Nc = numpy.count_nonzero(occupancy == 2)                    # Dimension of core space
    S = molecule.S

    half_density_matrix = S.dot(this.Total.Density/2).dot(S)
    NO_vals, NO_vects = numpy.linalg.eigh(half_density_matrix)  # See J. Chem. Phys. 1988, 88(8), 4926
    NO_coeffs = numpy.linalg.inv(S).dot(NO_vects)               # for details on finding the NO coefficents
    back_trans = numpy.linalg.inv(NO_coeffs)

    # Calculate the expectation value of the spin operator
    this.S2 = N*(N+4)/4. - Nab - 2 * sum([x ** 2 for x in NO_vals])   # Using formula from J. Chem. Phys. 88, 4926

    # Sort in order of descending occupancy
    idx = NO_vals.argsort()[::-1]     
    core_space = idx[:Nc]                            # Indices of the core NOs
    valence_space = idx[(Nc + Na):]                  # Indices of the valence NOs

    delta = (this.Alpha.Fock - this.Beta.Fock) / 2
    delta = NO_coeffs.T.dot(delta).dot(NO_coeffs)    # Transforming delta into the NO basis
    lambda_matrix = numpy.zeros(numpy.shape(delta))
    for i in core_space:
        for j in valence_space:
            lambda_matrix[i,j] = -delta[i,j]
            lambda_matrix[j,i] = -delta[j,i]
    lambda_matrix = back_trans.T.dot(lambda_matrix).dot(back_trans)  # Transforming lambda back to the AO basis

    this.Alpha.Fock = this.Alpha.Fock + lambda_matrix
    this.Beta.Fock = this.Beta.Fock - lambda_matrix

#----------------------------------------------------------------------

def level_shift(fock, shift_param, nElec, MOs):
    shift_matrix = numpy.zeros_like(fock)
    for i in range(nElec, len(shift_matrix)):
        shift_matrix[i,i] = shift_param
    
    #transform the matrix into the AO basis
    shift_matrix = MOs.dot(shift_matrix).dot(MOs.T)

    fock += shift_matrix

#----------------------------------------------------------------------

def print_intermediates(outfile, this, restricted):
    printf.text_value(outfile, " Alpha Orbital Energies ", this.Alpha.Energies, 
                      " Alpha MOs ", this.Alpha.MOs, 
                      " Alpha Density Matrix ", this.Alpha.Density) 
    if not restricted:
       printf.text_value(outfile, " Beta Orbital Energies ", this.Beta.Energies, 
                         " Beta MOs ", this.Beta.MOs, 
                         " Beta Density Matrix ", this.Beta.Density) 

def mix_alpha_beta(molecule, factor):
    molecule.Beta.MOs = factor * molecule.Alpha.MOs + (1 - factor) * molecule.Beta.MOs
    

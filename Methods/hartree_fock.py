# System libraries
import numpy
from numpy import dot
import scipy
from scipy.linalg import sqrtm

# Custom-written data modules
from Data import constants as c

# Custom-written utility modules
from Util import printf

# Custom-written code modules
import integrals
import hf_extensions as hf
#import initial_guess

#=================================================================#
#                                                                 #
#                        Main Function                            #
#                                                                 #
#=================================================================#

def do(settings, molecule, basis_set, state_index = 0):

    # Calculate values that are constant throughout the calculation
    molecule.NuclearRepulsion = integrals.nuclear_repulsion(molecule)
    make_core_matrices(molecule)

    # Set up for SCF calculation
    this_state = molecule.States[state_index]
    if state_index == 0:
       initialize_fock_matrices(molecule.Core, this_state)
       evaluate_2e_ints(molecule)

    # Generate initial orbitals and/or density matrices
    if this_state.Alpha.MOs != []:
        reference_orbitals = [this_state.Alpha.MOs, this_state.Beta.MOs]
    elif settings.SCF.Guess == "READ":
        try:
            this_state.Alpha.MOs = numpy.loadtxt(settings.AlphaMOFile)
            this_state.Beta.MOs = numpy.loadtxt(settings.BetaMOFile)
            assert len(this_state.Alpha.MOs) == molecule.NOrbitals
            assert len(this_state.Beta.MOs) == molecule.NOrbitals
            reference_orbitals = [this_state.Alpha.MOs, this_state.Beta.MOs]
        except:
            print('Error: Incorrect MOs supplied - check basis set and supply both alpha and beta sets')
            sys.exit()
    elif settings.SCF.Guess == "CORE":
        make_MOs(molecule, this_state)
        reference_orbitals = None
#    elif settings.SCF.Guess == "SAD":
#        molecule.AlphaDensity, molecule.BetaDensity = initial_guess.sad(molecule)
#        reference_orbitals = None

    if settings.SCF.Guess != "SAD":
        make_density_matrices(molecule, this_state)

    # Calculate initial energy
    calculate_energy(molecule, this_state)
    dE = this_state.Energy

    # Initial print
    if settings.PrintLevel == "VERBOSE":
       printf.delimited_text(settings.OutFile," Hartree-Fock initialization ")
       printf.text_value(settings.OutFile, " Nuclear repulsion energy: ", molecule.NuclearRepulsion)
       printf.text_value(settings.OutFile, " Alpha density matrix ", this_state.Alpha.Density,
                                           " Beta density matrix ", this_state.Beta.Density)
    printf.delimited_text(settings.OutFile," Hartree-Fock iterations ") 

    #-------------------------------------------#
    #           Begin SCF Iterations            #
    #-------------------------------------------#
    num_iterations = 0
    final_loop = False
    diis_error = None

    while c.energy_convergence < abs(dE):
        num_iterations += 1
        
        #-------------------------------------------#
        #               Main SCF step               #
        #-------------------------------------------#
#        initialize_fock_matrices(molecule.Core, this_state)
        make_coulomb_exchange_matrices(molecule, this_state)
        make_fock_matrices(molecule, this_state)
         
        # apply CUHF constraints
        if settings.SCF.Reference == "CUHF":
            constrain_UHF(molecule, this_state)

        #-------------------------------------------#
        #    Convergence accelerators/modifiers     #
        #-------------------------------------------#
        # DIIS
        if settings.DIIS.Use and num_iterations > 1:
            hf.diis.do(molecule, this_state, settings)
            diis_error = max(this_state.AlphaDIIS.Error, this_state.BetaDIIS.Error)

        # Update MOM reference orbitals to last iteration values if requested
        if settings.MOM.Use is True:
            if settings.MOM.Reference == 'MUTABLE':
                reference_orbitals = [this_state.Alpha.MOs, this_state.Beta.MOs]
        
        make_MOs(molecule, this_state)

        # Optionally, use MOM to reorder MOs
        if settings.MOM.Use and reference_orbitals != None:  
            hf.mom.do(molecule, this_state, state_index, reference_orbitals)
       #-------------------------------------------#

        make_density_matrices(molecule,this_state)

        old_energy = this_state.Energy
        calculate_energy(molecule, this_state)
        dE = this_state.Energy - old_energy
        this_state.TotalEnergy = this_state.Energy + molecule.NuclearRepulsion

        if abs(dE) < c.energy_convergence:
            final_loop = True

        # Loop print
        printf.text_value(settings.OutFile, " Cycle: ", num_iterations, " Total energy: ", this_state.TotalEnergy,
                          " Change in energy: ", dE, " DIIS error: ", diis_error)
        if settings.PrintLevel == "DEBUG":
           print_intermediates(settings.OutFile, this_state, (settings.SCF.Reference == "RHF"))

        if num_iterations >= settings.SCF.MaxIter:
            print("SCF not converging")
            break

    # Final print/dump to file
    printf.delimited_text(settings.OutFile, " End of Hartree-Fock iterations ")
    if this_state.S2 is not None: printf.text(settings.OutFile, "<S^2> = %.2f" % this_state.S2)
    printf.text_value(settings.OutFile, 'Final HF energy: ', this_state.TotalEnergy)
    printf.blank_line(settings.OutFile)
    print_intermediates(settings.OutFile, this_state, (settings.SCF.Reference == "RHF"))
    printf.delimited_text(settings.OutFile, " End of Hartree-Fock calculation ")
    if (settings.PrintLevel == 'VERBOSE'):
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

    energy = 0.0e0

    for a in range(0,molecule.NOrbitals):
       for b in range(0,molecule.NOrbitals):
          energy += 0.5e0*(this.Total.Density[a,b]*molecule.Core[a,b]+
                           this.Alpha.Density[a,b]*this.Alpha.Fock[a,b]+
                           this.Beta.Density[a,b] *this.Beta.Fock[a,b])

    this.Energy = energy

#----------------------------------------------------------------------

def make_core_matrices(molecule):

    for a in range(0,molecule.NCgtf):
      for b in range(a,molecule.NCgtf):
        shell_pair = molecule.ShellPairs[(a,b)]

        core,overlap = integrals.one_electron(molecule,shell_pair)

        na = shell_pair.Centre1.Cgtf.NAngMom
        nb = shell_pair.Centre2.Cgtf.NAngMom
        ia_vec = shell_pair.Centre1.Ivec
        ib_vec = shell_pair.Centre2.Ivec

        for i in range(0,na):
            for j in range(0,nb):
                molecule.Core[ia_vec[i]][ib_vec[j]] = core[i][j]
                molecule.Core[ib_vec[j]][ia_vec[i]] = core[i][j]
                molecule.Overlap[ia_vec[i]][ib_vec[j]] = overlap[i][j]
                molecule.Overlap[ib_vec[j]][ia_vec[i]] = overlap[i][j]

    # construct and store canonical orthogonalization matrices
    s,U = numpy.linalg.eigh(molecule.Overlap)
    sp = [element**-0.5e0 for element in s]
    molecule.X = numpy.dot(U,numpy.identity(len(sp))*(sp))
    molecule.Xt = numpy.transpose(molecule.X)
    # construct and store half-overlap matrix
    molecule.S = sqrtm(molecule.Overlap)

#----------------------------------------------------------------------

def evaluate_2e_ints(molecule,ints_type=0,grid_value=-1.0):

    for a in range(0,molecule.NCgtf):
      for b in range(0,molecule.NCgtf):
        for c in range(a,molecule.NCgtf):
          for d in range(0,molecule.NCgtf):

            if (b >= a):
              ab = molecule.ShellPairs[(a,b)]; 
              ia_vec = ab.Centre1.Ivec; ib_vec = ab.Centre2.Ivec
              nla = ab.Centre1.Cgtf.NAngMom; nlb = ab.Centre2.Cgtf.NAngMom  
              if (d >= c):
                cd = molecule.ShellPairs[(c,d)]; 
                ic_vec = cd.Centre1.Ivec; id_vec = cd.Centre2.Ivec
                nlc = cd.Centre1.Cgtf.NAngMom; nld = cd.Centre2.Cgtf.NAngMom

                coulomb = integrals.two_electron(ab,cd,ints_type,grid_value)

                for m in range(0,nla):
                    for n in range(0,nlb):
                        for l in range(0,nlc):
                            for s in range(0,nld):
                              molecule.CoulombIntegrals[ (ia_vec[m], ib_vec[n], ic_vec[l], id_vec[s]) ] = coulomb[m][n][l][s]
                              molecule.CoulombIntegrals[ (ib_vec[n], ia_vec[m], ic_vec[l], id_vec[s]) ] = coulomb[m][n][l][s]
                              molecule.CoulombIntegrals[ (ia_vec[m], ib_vec[n], id_vec[s], ic_vec[l]) ] = coulomb[m][n][l][s]
                              molecule.CoulombIntegrals[ (ib_vec[n], ia_vec[m], id_vec[s], ic_vec[l]) ] = coulomb[m][n][l][s]
                              molecule.CoulombIntegrals[ (ic_vec[l], id_vec[s], ia_vec[m], ib_vec[n]) ] = coulomb[m][n][l][s]
                              molecule.CoulombIntegrals[ (ic_vec[l], id_vec[s], ib_vec[n], ia_vec[m]) ] = coulomb[m][n][l][s]
                              molecule.CoulombIntegrals[ (id_vec[s], ic_vec[l], ia_vec[m], ib_vec[n]) ] = coulomb[m][n][l][s]
                              molecule.CoulombIntegrals[ (id_vec[s], ic_vec[l], ib_vec[n], ia_vec[m]) ] = coulomb[m][n][l][s]

            if (d >= a):
              ad = molecule.ShellPairs[(a,d)]; 
              ia_vec = ad.Centre1.Ivec; id_vec = ad.Centre2.Ivec
              nla = ad.Centre1.Cgtf.NAngMom; nld = ad.Centre2.Cgtf.NAngMom  
              if (b >= c):
                cb = molecule.ShellPairs[(c,b)]; 
                ic_vec = cb.Centre1.Ivec; ib_vec = cb.Centre2.Ivec
                nlc = cb.Centre1.Cgtf.NAngMom; nlb = cb.Centre2.Cgtf.NAngMom

                exchange = integrals.two_electron(ad,cb,ints_type,grid_value)

                for m in range(0,nla):
                    for n in range(0,nlb):
                        for l in range(0,nlc):
                            for s in range(0,nld):
                              molecule.ExchangeIntegrals[ (ia_vec[m], id_vec[s], ic_vec[l], ib_vec[n]) ] = exchange[m][s][l][n]
                              molecule.ExchangeIntegrals[ (id_vec[s], ia_vec[m], ic_vec[l], ib_vec[n]) ] = exchange[m][s][l][n]
                              molecule.ExchangeIntegrals[ (ia_vec[m], id_vec[s], ib_vec[n], ic_vec[l]) ] = exchange[m][s][l][n]
                              molecule.ExchangeIntegrals[ (id_vec[s], ia_vec[m], ib_vec[n], ic_vec[l]) ] = exchange[m][s][l][n]
                              molecule.ExchangeIntegrals[ (ic_vec[l], ib_vec[n], ia_vec[m], id_vec[s]) ] = exchange[m][s][l][n]
                              molecule.ExchangeIntegrals[ (ic_vec[l], ib_vec[n], id_vec[s], ia_vec[m]) ] = exchange[m][s][l][n]
                              molecule.ExchangeIntegrals[ (ib_vec[n], ic_vec[l], ia_vec[m], id_vec[s]) ] = exchange[m][s][l][n]
                              molecule.ExchangeIntegrals[ (ib_vec[n], ic_vec[l], id_vec[s], ia_vec[m]) ] = exchange[m][s][l][n]

#----------------------------------------------------------------------

def make_coulomb_exchange_matrices(molecule, this):
    
    this.Total.Coulomb.fill(0)
    this.Alpha.Exchange.fill(0)
    this.Beta.Exchange.fill(0)

    for a in range(0,molecule.NOrbitals):
      for b in range(0,molecule.NOrbitals):
        for c in range(0,molecule.NOrbitals):
          for d in range(0,molecule.NOrbitals):

             this.Total.Coulomb[a,b]  +=  this.Total.Density[c,d]*molecule.CoulombIntegrals[a,b,c,d]
             this.Alpha.Exchange[a,b] += -this.Alpha.Density[c,d]*molecule.ExchangeIntegrals[a,d,c,b]
             this.Beta.Exchange[a,b]  += -this.Beta.Density[c,d]*molecule.ExchangeIntegrals[a,d,c,b]

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
    NO_vals, NO_vects = numpy.linalg.eigh(half_density_matrix)  # See J. Chem. Phys. 88, 4926
    NO_coeffs = numpy.linalg.inv(S).dot(NO_vects)               # for details on finding the NO coefficents
    back_trans = numpy.linalg.inv(NO_coeffs)

    # Calculate the expectation value of the spin operator
    this.S2 = N*(N+4)/4. - Nab - 2 * sum([x ** 2 for x in NO_vals])   # Using formula from J. Chem. Phys. 88, 4926

    # Sort in order of descending occupancy
    idx = NO_vals.argsort()[::-1]                    # Note the [::-1] reverses the index array
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

def print_intermediates(outfile, this, restricted):
    printf.text_value(outfile, " Alpha Orbital Energies ", this.Alpha.Energies, 
                      " Alpha MOs ", this.Alpha.MOs, 
                      " Alpha Density Matrix ", this.Alpha.Density) 
    if not restricted:
       printf.text_value(outfile, " Beta Orbital Energies ", this.Beta.Energies, 
                         " Beta MOs ", this.Beta.MOs, 
                         " Beta Density Matrix ", this.Beta.Density) 

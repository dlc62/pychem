#This version has non of the OO refactoring done

import integrals
import Init
import constants as c
import copy
import numpy
from numpy import dot
from scipy.linalg import sqrtm
#import itertools as it
numpy.set_printoptions(precision = 5, linewidth = 100)  #Makes the arrays print nicely in the output 

########################### DIIS Functions ############################

def getDIISError(alpha_residuals, beta_residuals):
    max_alpha = abs(numpy.amax(alpha_residuals[-1]))
    max_beta = abs(numpy.amax(beta_residuals[-1]))
    error = max([max_alpha, max_beta])
#    print "DIIS Error"
#    print error 
    return error

def DoDIIS(residuals, focks):
    #rewrite this to calculate the  DIIS matrix efficently
    new_fock = numpy.zeros(numpy.shape(focks[0]))
    num = len(residuals) + 1
    DIIS_matrix = numpy.zeros((num,num))   
    DIIS_matrix[:,-1] = -1.
    DIIS_matrix[-1,:] = -1.
    DIIS_matrix[-1,-1] = 0.
    DIIS_vector = numpy.append(numpy.zeros((num-1,1)),[-1.]) 

    for i in xrange(len(residuals)):
        for j in xrange(len(residuals)):
            DIIS_matrix[i,j] = numpy.trace(residuals[i].dot(numpy.transpose(residuals[j])))
    
    rem = 0     #the number of residual vectors removed from the DIIS subspace 
    while numpy.linalg.cond(DIIS_matrix) > c.DIIS_MAX_CONDITION:
       #removing the rows and coloumns associated with the oldest error vectors 
       #until the condition of the matrix is acceptible.
       DIIS_matrix = numpy.delete(DIIS_matrix,0,0)
       DIIS_matrix = numpy.delete(DIIS_matrix,0,1)
       DIIS_vector = numpy.delete(DIIS_vector,0,0)  
       rem += 1

    print "DIIS Matrix"
    print DIIS_matrix
 
    coeffs = numpy.linalg.solve(DIIS_matrix, DIIS_vector) 

    for i in range(len(coeffs)-1):   # the last element in the coeffs vector is the Lagrange multiplyer
       new_fock += focks[i + rem] * coeffs[i]
   
    return new_fock


def getResidual(overlap, density, fock):
    residual  = overlap.dot(density).dot(fock) - fock.dot(density).dot(overlap)
   # transformed_residual = numpy.transpose(Xt).dot(residual).dot(X)
    return residual

########################## MOM and Excited State Functions  ########################### 

def maximumOverlapMethod(reference, new_MOs, energies, NElectrons, overlap_matrix):
    P_vector = numpy.zeros(len(new_MOs))
    dagger = numpy.transpose(reference[:,range(NElectrons)]) 
    MO_overlap = dagger.dot(overlap_matrix.dot(new_MOs))
    
    #sum the overlap of the individual AOs 
    if len(numpy.shape(MO_overlap)) == 1:
        P_vector = MO_overlap
    else: 
        for i in xrange(len(new_MOs)):
            P_vector[i] = sum(MO_overlap[:,i])
    
    sorted_MOs, sorted_energies = Sort_MOs(new_MOs, energies, numpy.abs(P_vector)) 
    return sorted_MOs, sorted_energies 

def Sort_MOs(MOs, energies,p):
#sorts MOs and energies in decending order 
#based on a vector p (the overlap vector)
    temp = [[p[i],MOs[:,i],energies[i]] for i in range(len(p))] 
    temp = sorted(temp, key = lambda temp: temp[0], reverse = True)     #sorts the elements on the basis of the p values (temp[0])
    new_MOs = numpy.array([line[1] for line in temp])
    new_energies = [line[2] for line in temp]
    return numpy.transpose(new_MOs), new_energies


def Excite(matrix,occupancy, NElectrons):
#This function permutes the an array give an list describing the
#orbital occupancy 
    new_matrix = copy.deepcopy(matrix)
    frm = []                        #list to contain the indexes orbitatals to be excited from
    to = []                         #list to contain the indexes of the orbitals to be excited to
    for i in range(NElectrons):
        if occupancy[i] == 0:
            frm.append(i)
    for i in range(NElectrons,len(occupancy)):
        if occupancy[i] == 1:
            to.append(i)
    for i in range(len(to)):
        new_matrix[:,[frm[i],to[i]]] = new_matrix[:,[to[i],frm[i]]]
    
    return new_matrix
            


########################### Basic HF Functions ############################

class ShellPair:
    def __init__(self,coords_a,cgtf_a,ia,ia_vec,coords_b,cgtf_b,ib,ib_vec):
       # precompute all ShellPair data
       # allocate list (or sparse array) for storing Schwarz bound integrals (and indices) for non-negligible values of the bound
       # because storing all bound values will get prohibitive (and silly!). 
       # note that ia & ib are vectors of Fock matrix indices for each cgtf of length NAngMom
       self.Centre1 = Shell(coords_a,cgtf_a,ia,ia_vec)
       self.Centre2 = Shell(coords_b,cgtf_b,ib,ib_vec)

class Shell:
    def __init__(self,coords,cgtf,index,index_vec):
       self.Coords = coords
       self.Cgtf = cgtf
       self.Index = index
       self.Ivec = index_vec
    
def make_MOs(X,Xt,fock_matrix):        
    transformed_fock_matrix = numpy.dot(Xt,numpy.dot(fock_matrix,X))    #orthoginalizing the fock matrix             
    orbital_energies,Cp = numpy.linalg.eigh(transformed_fock_matrix)    #solving the Roothan equations in the orthoginal basis 
    MOs = numpy.dot(X,Cp)                                               #transforming back into the orginal basis
    return MOs,orbital_energies
    
def make_density_matrix(density_matrix,MOs,NElectrons):
    # Construct one-electron density matrix
    n_basis_functions = len(density_matrix)
    for ia in range(0,n_basis_functions):
        for ib in range(0,n_basis_functions):
           density_matrix[ia][ib] = 0.0e0
           for n in range(0,NElectrons):
               density_matrix[ia][ib] += MOs[ia][n]*MOs[ib][n]
    return density_matrix

def calculate_energy(alpha_density_matrix,beta_density_matrix,total_density_matrix,alpha_fock_matrix,beta_fock_matrix,core_fock_matrix):
    n_basis_functions = len(alpha_density_matrix)
    energy = 0.0e0
    for mu in range(0,n_basis_functions):
       for nu in range(0,n_basis_functions):
          energy += 0.5e0*(total_density_matrix[mu][nu]*core_fock_matrix[mu][nu]+
                           alpha_density_matrix[mu][nu]*alpha_fock_matrix[mu][nu]+
                           beta_density_matrix[mu][nu]*beta_fock_matrix[mu][nu]) 
    return energy

def makeTemplateMatrix(n):
    template_matrix_row = [0.0 for i in range(0,n)]      
    return numpy.array([copy.deepcopy(template_matrix_row) for i in range(0,n)])

###################################################################
#                                                                 #
#                        Main Function                            #
#                                                                 #
###################################################################

def do(system,molecule,state,alpha_reference, beta_reference):      
    num_iterations = 0
    isFirstCalc =(alpha_reference[0][0] == None)  

    #variables for DIIS 
    if system.UseDIIS == True:
        old_alpha_focks = []
        old_beta_focks = []
        alpha_residuals = []
        beta_residuals = []
        total_residuals = []
        DIIS_error = 1     
##      limiting the DIIS subspace size for small moleucle to prevent linear dependence
#       if molecule.NOrbitals < system.DIISSize:
#            system.DIISSize = molecule.NOrbitals
    
    nuclear_repulsion_energy = integrals.nuclear_repulsion(molecule)
    # initialize Fock matrix
    n_basis_functions = 0           #total number of basis functions
    n_shells = 0
    for atom in molecule.Atoms:
       for cgtf in atom.Basis:
          n_basis_functions += cgtf.NAngMom
          n_shells += 1
    #creating a series of square matrices with rank equal to the number of basis functions.
    template_matrix = makeTemplateMatrix(n_basis_functions) 
    core_fock_matrix = copy.deepcopy(template_matrix)
#    delta_fock_matrix = copy.deepcopy(template_matrix)
    overlap_matrix = numpy.array(copy.deepcopy(template_matrix))
#    schwarz_bound = copy.deepcopy(template_matrix)
    alpha_density_matrix = numpy.array(copy.deepcopy(template_matrix))
    beta_density_matrix = numpy.array(copy.deepcopy(template_matrix))
    # form core Fock matrix and overlap matrix once and for all, setting up shell pairs as we go 
    ia = -1
    ia_count = 0
    shell_pairs = []
    for atom_a in molecule.Atoms:
       for cgtf_a in atom_a.Basis:
          ia += 1
          ia_vec = [(ia_count + i) for i in range(0,cgtf_a.NAngMom)]   #vector contaning the indices of the orbitals 
          ia_count += cgtf_a.NAngMom                                   #total number of orbitals
          ib_count = 0
          ib = -1
          for atom_b in molecule.Atoms:
             for cgtf_b in atom_b.Basis:
                ib_vec = [(ib_count + i) for i in range(0,cgtf_b.NAngMom)]
                ib_count += cgtf_b.NAngMom
                shell_pair = ShellPair(atom_a.Coordinates,cgtf_a,ia,ia_vec,atom_b.Coordinates,cgtf_b,ib,ib_vec)   #forming all possible shell pairs 
                shell_pairs.append(shell_pair)
                core,overlap = integrals.one_electron(molecule,shell_pair)        #core-interaction and overlap integrals for the shell pair
                for i in range(0,cgtf_a.NAngMom):
                   for j in range(0,cgtf_b.NAngMom): 
                      core_fock_matrix[ia_vec[i]][ib_vec[j]] = core[i][j]         #Constructing the core-fock and overlap matrices 
                      overlap_matrix[ia_vec[i]][ib_vec[j]] = overlap[i][j]
#                print 'core_fock_matrix', core_fock_matrix
#                print 'overlap_matrix', overlap_matrix
##     uncomment these print statements 
#    print '****************************************************'
#    print ' Initialization '
#    print '****************************************************'
#    print 'Nuclear repulsion energy', nuclear_repulsion_energy
#    print 'Core Fock matrix'
#    print core_fock_matrix
    ## if reference_orbitals == None, do diagonalization to get initial MOs
    ## MOs are mutable but reference orbitals don't change during HF procedure
    s,U = numpy.linalg.eigh(overlap_matrix)
    sp = [element**-0.5e0 for element in s]
    X = numpy.dot(U,numpy.identity(len(sp))*(sp))
    Xt = numpy.transpose(X)
    
    #Generating the initial density matrices 
    if isFirstCalc == False:
        alpha_MOs, beta_MOs, alpha_density_matrix, beta_density_matrix = Init.readGuess(alpha_reference, beta_reference, state, molecule)
        alpha_reference = copy.deepcopy(alpha_MOs)
        beta_reference = copy.deepcopy(beta_MOs)
    elif system.SCFGuess == 'core':
        alpha_MOs, beta_MOs, alpha_density_matrix, beta_density_matrix = Init.coreGuess(core_fock_matrix, X, Xt, molecule)
    elif system.SCFGuess == 'sad':
        alpha_density_matrix, beta_density_matrix = Init.sadGuess(molecule, system.BasisSets[0])
        alpha_MOs = copy.deepcopy(template_matrix)
        beta_MOs = copy.deepcopy(template_matrix)
    total_density_matrix = numpy.ndarray.tolist(numpy.add(alpha_density_matrix,beta_density_matrix))
   
    energy = calculate_energy(alpha_density_matrix,beta_density_matrix,total_density_matrix,
                              core_fock_matrix,core_fock_matrix,core_fock_matrix)
    dE = energy 
    DIIS_error = 0     #set to zero so the convergence criterta is met when not using DIIS
#    print 'Guess alpha density matrix'
#    print alpha_density_matrix
#    print '****************************************************'
#    print ' Hartree-Fock iterations '
#    print '****************************************************'
   

################################################
#              Begin Iterations                # 
################################################

    while (abs(dE) > c.energy_convergence):
               
       num_iterations += 1
       alpha_fock_matrix = copy.deepcopy(core_fock_matrix)
       beta_fock_matrix = copy.deepcopy(core_fock_matrix)
       coulomb_matrix = copy.deepcopy(template_matrix)
       alpha_exchange_matrix = copy.deepcopy(template_matrix)
       beta_exchange_matrix = copy.deepcopy(template_matrix)
       old_energy = copy.copy(energy)
#       old_alpha_density_matrix = copy.deepcopy(alpha_density_matrix)
#       old_beta_density_matrix = copy.deepcopy(beta_density_matrix)
#       screen = numpy.zeroes((shell_pair1.Centre1.Cgtf.NAngMom,\
#                              shell_pair1.Centre2.Cgtf.NAngMom,\
#                              shell_pair2.Centre1.Cgtf.NAngMom,\
#                              shell_pair2.Centre2.Cgtf.NAngMom)
       # Use guess/current MOs (supplied or newly calculated) to calculate two-electron contributions to Fock matrix
       for shell_pair1 in shell_pairs:
          ia_vec = shell_pair1.Centre1.Ivec
          ib_vec = shell_pair1.Centre2.Ivec
          for shell_pair2 in shell_pairs:
             ic_vec = shell_pair2.Centre1.Ivec
             id_vec = shell_pair2.Centre2.Ivec
#             for m in range(0,shell_pair1.Centre1.Cgtf.NAngMom):
#                for n in range(0,shell_pair1.Centre2.Cgtf.NAngMom):
#                   for l in range(0,shell_pair2.Centre1.Cgtf.NAngMom):
#                      for s in range(0,shell_pair2.Centre2.Cgtf.NAngMom):
#                         screen[m,n,l,s] = overlap_matrix[ia_vec[m]][ib_vec[n]]*overlap_matrix[ic_vec[l]][id_vec[s]]
             coulomb,exchange = integrals.two_electron(shell_pair1,shell_pair2)
             for m in range(0,shell_pair1.Centre1.Cgtf.NAngMom):
                for n in range(0,shell_pair1.Centre2.Cgtf.NAngMom):
                   for l in range(0,shell_pair2.Centre1.Cgtf.NAngMom):
                      for s in range(0,shell_pair2.Centre2.Cgtf.NAngMom):
                         coulomb_matrix[ia_vec[m]][ib_vec[n]] += total_density_matrix[ic_vec[l]][id_vec[s]]*coulomb[m][n][l][s]
                         alpha_exchange_matrix[ia_vec[m]][ib_vec[n]] += -alpha_density_matrix[ic_vec[l]][id_vec[s]]*exchange[m][s][l][n]
                         beta_exchange_matrix[ia_vec[m]][ib_vec[n]] += -beta_density_matrix[ic_vec[l]][id_vec[s]]*exchange[m][s][l][n]
          for m in range(0,shell_pair1.Centre1.Cgtf.NAngMom):
             for n in range(0,shell_pair1.Centre2.Cgtf.NAngMom):
                alpha_fock_matrix[ia_vec[m]][ib_vec[n]] += (coulomb_matrix[ia_vec[m]][ib_vec[n]] + alpha_exchange_matrix[ia_vec[m]][ib_vec[n]])
                beta_fock_matrix[ia_vec[m]][ib_vec[n]] += (coulomb_matrix[ia_vec[m]][ib_vec[n]] + beta_exchange_matrix[ia_vec[m]][ib_vec[n]])
              
           
###### Constrained UHF stuff follows (only for Multiplicity != 1)
       if molecule.Multiplicity != 1:
           overlap_sqrt_matrix = numpy.matrix(sqrtm(overlap_matrix))
           scaled_density_matrix = overlap_sqrt_matrix*numpy.matrix(total_density_matrix)*overlap_sqrt_matrix
           NO_test_matrix = numpy.absolute(numpy.subtract(scaled_density_matrix,numpy.identity(len(scaled_density_matrix))))
           if numpy.amax(NO_test_matrix) > c.integral_threshold:
               NO_occs,scaled_NOs = numpy.linalg.eigh(scaled_density_matrix)
               NOs = numpy.linalg.solve(overlap_sqrt_matrix,scaled_NOs)
               idx = NO_occs.argsort()[::-1]; sorted_NO_occs = NO_occs[idx]; sorted_NOs = NOs[:,idx]
               if ((molecule.NAlphaElectrons != molecule.NOrbitals) and (molecule.NBetaElectrons != 0)):
                   delta_fock_matrix = numpy.ndarray.tolist(sorted_NOs*numpy.subtract(alpha_fock_matrix,beta_fock_matrix)/2)
                   for i in range(0,molecule.NBetaElectrons):
                       for j in range(molecule.NAlphaElectrons,molecule.NOrbitals):
                           delta_fock_matrix[i][j] = -delta_fock_matrix[i][j]
                           delta_fock_matrix[j][i] = -delta_fock_matrix[j][i]
                   lambda_fock_matrix = numpy.linalg.inv(sorted_NOs)*numpy.matrix(delta_fock_matrix)
                   constrained_alpha_fock_matrix = numpy.matrix(alpha_fock_matrix) + lambda_fock_matrix 
                   constrained_beta_fock_matrix = numpy.matrix(alpha_fock_matrix) - lambda_fock_matrix 
                   alpha_fock_matrix = numpy.ndarray.tolist(constrained_alpha_fock_matrix)
                   beta_fock_matrix = numpy.ndarray.tolist(constrained_beta_fock_matrix)
#           else
#               MOs are NOs, and occupancies are as specified
                   
################################### Start DIIS ############################################################
      
      #preforming DIIS starting from the first variational 
       if num_iterations > 1 and system.UseDIIS == True:
           
           #limiting the size of the DIIS subsapce 
           if len(alpha_residuals) >= system.DIISSize:
               alpha_residuals.pop(0)
               beta_residuals.pop(0)
               total_residuals.pop(0)
               old_alpha_focks.pop(0)
               old_beta_focks.pop(0)
           

           alpha_residuals.append(getResidual(overlap_matrix, alpha_density_matrix, alpha_fock_matrix))
           beta_residuals.append(getResidual(overlap_matrix, beta_density_matrix, beta_fock_matrix))
           #combing the two residual vectors, the Qchem manual implies this is an approximation 
           #but this code doesn't seem to work without it.
           total_residuals.append(alpha_residuals[-1] + beta_residuals[-1])
           DIIS_error = numpy.amax(total_residuals[-1])
           old_alpha_focks.append(copy.deepcopy(alpha_fock_matrix))
           old_beta_focks.append(copy.deepcopy(beta_fock_matrix))
           old_DIIS_error = DIIS_error
           print "DIIS Error"
           print DIIS_error 
           if len(alpha_residuals) > 1: 
#              Preforming a DIIS step and storing the extrapolated fock matrix
#              Note this alos introduces a very small chance of the alpha and beta matrices cancelling out.
               alpha_fock_matrix = DoDIIS(total_residuals, old_alpha_focks)
               beta_fock_matrix = DoDIIS(total_residuals, old_beta_focks)
               old_alpha_focks[-1] = copy.deepcopy(alpha_fock_matrix)
               old_beta_focks[-1]  = copy.deepcopy(beta_fock_matrix)                                                                      
                                                                        
############################################ End DIIS ################################################################                   
    
       #Taking a copy of the old MOS to use as a reference for the MOM
       if system.MOM_Type != "fixed":
         alpha_reference = copy.deepcopy(alpha_MOs)
         beta_reference = copy.deepcopy(beta_MOs)

       alpha_MOs, alpha_orbital_energies = make_MOs(X,Xt,alpha_fock_matrix)
       beta_MOs, beta_orbital_energies = make_MOs(X,Xt,beta_fock_matrix) 
       
#       print "Orbital Energies Pre-Sorting"
#       print alpha_orbital_energies 
                  

#################### Maximum Overlap Method ##########################

       if system.UseMOM == True and isFirstCalc == False: 
           alpha_MOs, alpha_orbital_energies = maximumOverlapMethod(alpha_reference, alpha_MOs, alpha_orbital_energies,
                                                                    molecule.NAlphaElectrons, overlap_matrix)
           beta_MOs, beta_orbital_energies = maximumOverlapMethod(beta_reference, beta_MOs, beta_orbital_energies,
                                                                    molecule.NBetaElectrons, overlap_matrix)

#################### End  Maximum Overlap Method #####################

       alpha_density_matrix = make_density_matrix(alpha_density_matrix,alpha_MOs,molecule.NAlphaElectrons)
       beta_density_matrix = make_density_matrix(beta_density_matrix,beta_MOs,molecule.NBetaElectrons)
       total_density_matrix = numpy.ndarray.tolist(numpy.add(alpha_density_matrix,beta_density_matrix))            
       
       energy = calculate_energy(alpha_density_matrix,beta_density_matrix,total_density_matrix,alpha_fock_matrix,beta_fock_matrix,core_fock_matrix)
       dE = energy - old_energy
              
#       print "Overlap Matrix"
#       print alpha_O
#       print "Overlap vector"
#       print alpha_p
##       print "Orbital Energies Post Sorting"
##       print alpha_orbital_energies
#       print "MOs Pre Sorting" 
#       print old_alpha_MOs 
#       print "MOs Post Sorting"
#       print alpha_MOs
  
       print "Cycle: " , num_iterations
       print 'Total energy'
       print energy + nuclear_repulsion_energy
#       print 'Coulomb matrix'
#       print coulomb_matrix
#       print 'Alpha exchange matrix'
#       print alpha_exchange_matrix
#       print 'Alpha Fock matrix'
#       print alpha_fock_matrix
       print 'Alpha orbital energies'
       print alpha_orbital_energies
       print 'Alpha MO coefficients'
       print alpha_MOs
       print 'Alpha density matrix'
       print alpha_density_matrix
       print "Beta MO coeficents"
       print beta_MOs
#       print "Beta Density Matrix"
#       print beta_density_matrix
       print "Beta orbital energies"
       print beta_orbital_energies
       print '----------------------------------------------------'
    print '                       End                          '
    print '----------------------------------------------------'
    return alpha_MOs, beta_MOs


####################################################

    

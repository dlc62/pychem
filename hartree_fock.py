import integrals
import Init 
import constants as c
import copy
import numpy 
from numpy import dot 
from scipy.linalg import sqrtm
numpy.set_printoptions(precision = 5, linewidth = 100)  #Makes the arrays print nicely in the output 

#the MOM and excited state functions haven't made it to the version  

########################### DIIS Object ############################

class DIIS_system:
    def __init__(self, max_condition,max_space):
        self.oldAlphaFocks = []
        self.oldBetaFocks = []
        self.matrix = None  
        self.error = 1
        self.alphaResiduals = [] 
        self.betaResiduals = []
        self.residuals = []       #total residuals
        self.max_cond = max_condition
        self.max_space = max_space

    def getDIISError(self, Print = False):
        max_alpha = abs(numpy.amax(self.alphaResiduals[-1]))
        max_beta = abs(numpy.amax(self.betaResiduals[-1]))
        self.error = max([max_alpha, max_beta])
        if Print == True:
            print "DIIS Error"
            print self.error 

    def getResidual(self, overlap, density, fock):
        residual  = overlap.dot(density).dot(fock) - fock.dot(density).dot(overlap)
       # transformed_residual = numpy.transpose(Xt).dot(residual).dot(X)
        return residual

    def innerProd(matA,matB):
    #takes two matrices and returns the trace of the product 
        product = self.matA.dot(transpose(self.matB))
        return numpy.trace(product)

    def makeDISSMatrix(self):
        #generating the first DIIS matrix 
        if self.matrix == None:
            self.matrix = numpy.zeros((3,3))   #assuming DIIS starts one two error vectors have been found  
            for i in xrange(len(self.totalResiduals)):
                for j in xrange(len(self.residuals)):
                    self.matrix[i,j] = innerProd(self.residuals[i], self.residuals[j])
            self.matrix[2:0:2] = -1.
            self.matrix[2,0:2] = -1.
        #extending an old DIIS matrix 
        else:
            size = len(self.matrix)
            #updating the old DIIS_Matrix with the new exrapolated fock matrix 
            #from the prvious DIIS step
            new_vector = numpy.zeros(size-1)
            for i in xrange(len(new_vector)):
                new_vector[i] += innerProd(self.residuals[-2], self.residuals[i])
            self.matrix[size-2,0:size-1] = new_vector
            self.matrix[0:size-1,size-2] = numpy.transpose(new_vector)

            #making the new row of the matrix 
            new_matrix = zeros((size+1,size+1))
            new_matrix[0:size,0:size] = self.matrix
            new_vector = numpy.zeros(size)
            for i in xrange(size):
                new_vector[i] = innerProd(self.residuals[-1], self.residuals[i])
            new_matrix[size-1,0:size] = new_vector
            new_matrix[0:size,size-1] = numpy.transpose(new_vector)
            new_matrix[size, 0:size] = -1.
            new_matrix[0:size, size] = -1.
        self.matrix = new_matrix

    def reduceSpace(self):
    #reduces the size of the fock space by one 
    #removes the oldest element
        self.matrix = numpy.delete(self.matrix,0,0)
        self.matrix = numpy.delete(self.matrix,0,1)
        self.oldAlphaFocks.pop(0) 
        self.oldBetaFocks.pop(0)
        self.alphaResiduals.pop(0) 
        self.betaResiduals.pop(0)
        self.residuals.pop(0)
    
    def makeFock(self, focks):
        new_focks = numpy.zeros(numpy.shape(focks[0]))
        size = len(self.matrix)
        DIIS_vector = numpy.append(numpy.zeros((size-1,1)),[-1.]) 
        coeffs = numpy.linalg.solve(self.matrix,DIIS_vector)
        for i in xrange(size):
            new_fock += focks[i] * coeffs[i]
        return new_focks
    
    def updateDIIS(self, alpha_fock, beta_fock, densities):
    #generates takes fock matrices and generates a new set of residuals 
    #plus monitors the size of the DIIS space
        if len(self.alphaResiduals >= self.max_space):
            self.reduceSpace()
        self.alphaResidals.append(self.getResiduals(overlap_matrix, densities.alpha, alpha_fock_matrix))
        self.betaResidals.append(self.getResiduals(overlap_matrix, densities.beta, beta_fock_matrix))     
        self.residuals.append(self.alphaResiduals[-1] + self.betaResiduals[-1]) 

    def storeExtrapolated(self,alpha_fock, beta_fock, densities):
        self.oldAlphaFocks[-1] = copy.deepcopy(alpha_fock)
        self.oldBeatFocks[-1] = copy.deepcopy(beta_fock)
        self.alphaResiduals[-1] = self.getResiduals(overlap_matrix, densities.alpha, alpha_fock)
        self.betaResiduals[-1] = self.getResiduals(overlap_matrix, densities.beta, beta_fock)
        self.residuals = dd(self.alphaResiduals + self.betaResiduals)
            

    def DoDIIS(self, focks, densities):
        self.updateDIIS(focks.alpha, focks.beta, densities)
        self.matrix = self.makeDIISMatrix()
        while numpy.linalg.cond(self.matrix) > self.max_cond:
            self = self.reduceSpace()
        #making the new fock matrices 
        alpha_fock = self.makeFock(self.oldAlphaFocks)
        beta_fock = self.makeFock(self.oldBetaFocks)
        self.getDIISError(Print = True)
        #replcing the information from the old fock matrices with ath 
        #from the extrapolated matrices 
        self.storeExtrapolated(alpha_fock, beta_fock, densities)
        return alpha_fock, beta_fock 


########################### Basic HF Functions ############################

class Density_matrix:
    def __init__(self, alpha_matrix, beta_matrix):
        alpha = alpha_matrix
        beta = beta_matrix
        total = alpha_matrix + beta_matrix 

class Fock_matrix:
    def __init__(self):
        core = []
        alpha = []
        beta = []

    def resetFocks(self):
    #sets the alpha and beta fock matrcies as the core 
        self.alpha = copy.deepcopy(self.core)
        self.beta = copy.deepcopy(self.core)

def makeFockMatrices(self, densities, shell_pairs,template_matrix):
       self.resetFocks() 
       coulomb_matrix = copy.deepcopy(template_matrix)
       alpha_exchange_matrix = copy.deepcopy(template_matrix)
       beta_exchange_matrix = copy.deepcopy(template_matrix)
#      old_alpha_density_matrix = copy.deepcopy(alpha_density_matrix)
#      old_beta_density_matrix = copy.deepcopy(beta_density_matrix)
#      screen = numpy.zeroes((shell_pair1.Centre1.Cgtf.NAngMom,\
#                             shell_pair1.Centre2.Cgtf.NAngMom,\
#                             shell_pair2.Centre1.Cgtf.NAngMom,\
#                             shell_pair2.Centre2.Cgtf.NAngMom)
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
                         coulomb_matrix[ia_vec[m]][ib_vec[n]] += densities.total[ic_vec[l]][id_vec[s]]*coulomb[m][n][l][s]
                         alpha_exchange_matrix[ia_vec[m]][ib_vec[n]] += -densities.alpha[ic_vec[l]][id_vec[s]]*exchange[m][s][l][n]
                         beta_exchange_matrix[ia_vec[m]][ib_vec[n]] += -densities.beta[ic_vec[l]][id_vec[s]]*exchange[m][s][l][n]
          for m in range(0,shell_pair1.Centre1.Cgtf.NAngMom):
             for n in range(0,shell_pair1.Centre2.Cgtf.NAngMom):
                self.alpha[ia_vec[m]][ib_vec[n]] += (coulomb_matrix[ia_vec[m]][ib_vec[n]] + alpha_exchange_matrix[ia_vec[m]][ib_vec[n]])
                self.beta[ia_vec[m]][ib_vec[n]] += (coulomb_matrix[ia_vec[m]][ib_vec[n]] + beta_exchange_matrix[ia_vec[m]][ib_vec[n]])


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

def calculate_energy(fock, density):
    n_basis_functions = len(density.alpha)
    energy = 0.0e0
    for mu in range(0,n_basis_functions):
       for nu in range(0,n_basis_functions):
          energy += 0.5e0*(density.total[mu][nu]*fock.core[mu][nu]+
                           density.alpha[mu][nu]*fock.alpha[mu][nu]+
                           density.beta[mu][nu]*fock.beta[mu][nu]) 
    return energy

def makeTemplateMatrix(n):
    template_matrix_row = [0.0 for i in range(0,n)]      
    return numpy.array([copy.deepcopy(template_matrix_row) for i in range(0,n)])

def makeCoreMatrices(template_matrix, molecule):
    core_fock_matrix = copy.deepcopy(template_matrix)
    overlap_matrix = numpy.array(copy.deepcopy(template_matrix))
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
    return core_fock_matrix, overlap_matrix, shell_pairs 

def constrainedUHF(overlap_matrix, density, molecule, fock):
   overlap_sqrt_matrix = numpy.matrix(sqrtm(overlap_matrix))
   scaled_density_matrix = overlap_sqrt_matrix*numpy.matrix(density.total)*overlap_sqrt_matrix
   NO_test_matrix = numpy.absolute(numpy.subtract(scaled_density_matrix,numpy.identity(len(scaled_density_matrix))))
   if numpy.amax(NO_test_matrix) > c.integral_threshold:
       NO_occs,scaled_NOs = numpy.linalg.eigh(scaled_density_matrix)
       NOs = numpy.linalg.solve(overlap_sqrt_matrix,scaled_NOs)
       idx = NO_occs.argsort()[::-1]; sorted_NO_occs = NO_occs[idx]; sorted_NOs = NOs[:,idx]
       if ((molecule.NAlphaElectrons != molecule.NOrbitals) and (molecule.NBetaElectrons != 0)):
           delta_fock_matrix = numpy.ndarray.tolist(sorted_NOs*numpy.subtract(fock.alpha,fock.beta)/2)
           for i in range(0,molecule.NBetaElectrons):
               for j in range(molecule.NAlphaElectrons,molecule.NOrbitals):
                   delta_fock_matrix[i][j] = -delta_fock_matrix[i][j]
                   delta_fock_matrix[j][i] = -delta_fock_matrix[j][i]
           lambda_fock_matrix = numpy.linalg.inv(sorted_NOs)*numpy.matrix(delta_fock_matrix)
           constrained_alpha_fock_matrix = numpy.matrix(alpha_fock_matrix) + lambda_fock_matrix 
           constrained_beta_fock_matrix = numpy.matrix(alpha_fock_matrix) - lambda_fock_matrix 
           alpha_fock_matrix = numpy.ndarray.tolist(constrained_alpha_fock_matrix)
           beta_fock_matrix = numpy.ndarray.tolist(constrained_beta_fock_matrix)
   return alpha_fock_matrix, beta_fock_matrix 

###################################################################
#                                                                 #
#                        Main Function                            #
#                                                                 #
###################################################################

def do(system, state, molecule, alpha_reference, beta_reference):
    num_iterations = 0
    isFirstCalc = (alpha_reference[0][0] == None)
    fock = Fock_matrix() 
    density = Desnity_matrix()
    DIIS = DIISsystem(c.DIIS_MAX_CONVERGENCE, system.DIIS_size)
    
    #setting up the values that are constant througout the calculation
    nuclear_repulsion_energy = integrals.nuclear_repulsion(molecule)
    template_matrix =  makeTemplateMatrix(molecule.NOrbitals)    
    fock.core, overlap, shell_pairs = makeCoreMatrices(template_matrix, molecule) 
    fock.resetFocks()

#    print '****************************************************'
#    print ' Initialization '
#    print '****************************************************'
#    print 'Nuclear repulsion energy', nuclear_repulsion_energy
#    print 'Core Fock matrix'
#    print core_fock_matrix

    s,U = numpy.linalg.eigh(overlap_matrix)
    sp = [element**-0.5e0 for element in s]
    X = numpy.dot(U,numpy.identity(len(sp))*(sp))
    Xt = numpy.transpose(X)

    #Generating the initial density matrices 
    #Note there are no reference orbitals in the first caclulation 
    if isFirstCalc == False:
        alpha_MOs, beta_MOs, density = Init.readGuess(alpha_reference, beta_reference, state, molecule)
        alpha_reference = copy.deepcopy(alpha_MOs)
        beta_reference = copy.deepcopy(beta_MOs)
    elif system.SCFGuess == 'core':
        alpha_MOs, beta_MOs, density = Init.coreGuess(fock.core, X, Xt, molecule)
    elif system.SCFGuess == 'sad':
        density = Init.sadGuess(molecule, system.BasisSets[0])
        alpha_MOs = copy.deepcopy(template_matrix)
        beta_MOs = copy.deepcopy(template_matrix)
    
    energy = calculate_energy(density, fock)
    dE = energy 
    if system.UseDIIS != True:
        DIIS.error = 0     #set to zero so the convergence criterta is met when not using DIIS

#    print 'Guess alpha density matrix'
#    print alpha_density_matrix
#    print '****************************************************'
#    print ' Hartree-Fock iterations '
#    print '****************************************************'


    while (abs(dE)) > c.energy_convergence):
        num_iterations += 1 
        fock.makeFockMatrices(density, sehell_pairs, template_matrix) 
       
        if molecule.multiplicity != 1:
           fock.alpha, fock.beta = constrainedUHF(overlap_matrix, density, molecule, focks)

       #preforming DIIS starting from the first variational fock matrix 
        if system.UseDIIS == True and num_iterations > 1: 
           fock.alpha, fock.beta = DIIS.DoDIIS(self, fock, density)

        if system.MOM_Type != "fixed":
            alpha_reference = copy.deepcopy(alpha_MOs)
            beta_reference = copy.deepcopy(beta_MOs)
        
        alpha_MOs, alpha_orbital_energies = make_MOs(X,Xt,fock.alpha)
        beta_MOs, beta_orbital_energies = make_MOs(X,Xt,fock.beta)

       #Maximum Overlap Method 
        if system.UseMOM == True and isFirstCalc == False: 
            alpha_MOs, alpha_orbital_energies = maximumOverlapMethod(alpha_reference, alpha_MOs, alpha_orbital_energies,
                                                                    molecule.NAlphaElectrons, overlap_matrix)
            beta_MOs, beta_orbital_energies = maximumOverlapMethod(beta_reference, beta_MOs, beta_orbital_energies,
                                                                    molecule.NBetaElectrons, overlap_matrix

        density.alpha = make_density_matrix(density.alpha,alpha_MOs,molecule.NAlphaElectrons)
        density.beta = make_density_matrix(density.beta,beta_MOs,molecule.NBetaElectrons)
        density.total = numpy.ndarray.tolist(numpy.add(density.alpha,density.beta))            
       
        energy = calculate_energy(density, fock)
        dE = energy - old_energy
              
#        print "Overlap Matrix"
#        print alpha_O
#        print "Overlap vector"
#        print alpha_p
##        print "Orbital Energies Post Sorting"
##        print alpha_orbital_energies
#        print "MOs Pre Sorting" 
#        print old_alpha_MOs 
#        print "MOs Post Sorting"
#        print alpha_MOs
  
        print "Cycle: " , num_iterations
        print 'Total energy'
        print energy + nuclear_repulsion_energy
#Will need to get the exchange matrices out of the makeFockMatrix function  
#        print 'Coulomb matrix'
#        print coulomb_matrix
#        print 'Alpha exchange matrix'
#        print alpha_exchange_matrix
#        print 'Alpha Fock matrix'
#        print alpha_fock_matrix
        print 'Alpha orbital energies'
        print alpha_orbital_energies
        print 'Alpha MO coefficients'
        print alpha_MOs
        print 'Alpha density matrix'
        print alpha_density_matrix
        print "Beta MO coeficents"
        print beta_MOs
#        print "Beta Density Matrix"
#        print beta_density_matrix
        print "Beta orbital energies"
        print beta_orbital_energies
        print '----------------------------------------------------'
    print '                       End                          '
    print '----------------------------------------------------'
    return alpha_MOs, beta_MOs


####################################################

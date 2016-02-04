import integrals
import Init 
import constants as c
import copy
import numpy 
import Output
from numpy import dot 
from scipy.linalg import sqrtm
numpy.set_printoptions(precision = 5, linewidth = 100)  #Makes the arrays print nicely in the output 

########################### DIIS Object ############################

class DIIS_system:
    def __init__(self, max_condition,max_space, DIIS_Type):
        self.oldAlphaFocks = []
        self.oldBetaFocks = []
        self.matrix = None  
        self.error = 1
        self.alphaResiduals = [] 
        self.betaResiduals = []
        self.residuals = []       #total residuals
        self.DIIS_Type = DIIS_Type  
        self.max_cond = max_condition
        self.max_space = max_space

    def getDIISError(self):
        max_alpha = abs(numpy.amax(self.alphaResiduals[-1]))
        max_beta = abs(numpy.amax(self.betaResiduals[-1]))
        self.error = max([max_alpha, max_beta])

    def getResidual(self, overlap, density, fock):
        residual  = overlap.dot(density).dot(fock) - fock.dot(density).dot(overlap)
       # transformed_residual = numpy.transpose(Xt).dot(residual).dot(X)
        return residual

    def innerProd(self, matA,matB):
    #takes two matrices and returns the trace of the product 
        product = matA.dot(numpy.transpose(matB))
        return numpy.trace(product)

    def makeDIISMatrix(self):
        #generating the first DIIS matrix 
        if self.matrix == None:
            self.matrix = numpy.zeros((3,3))   #assuming DIIS starts once two error vectors have been found  
            for i in xrange(len(self.residuals)):
                for j in xrange(len(self.residuals)):
                    self.matrix[i,j] = self.innerProd(self.residuals[i], self.residuals[j])
            # setting up the parts of the DIIS matrix particular to C1 and C2 DIIS 
            if self.DIIS_Type == "C1":
                self.matrix[0:2,2] = -1.
                self.matrix[2,0:2] = -1.
            else:
                self.matrix = self.matrix[0:-1,0:-1]
        #extending an old DIIS matrix 
        else:
            size = len(self.matrix)
            # updating the old DIIS_Matrix with the extrapolated fock matrix 
            # from the prvious DIIS step
            update_vector= numpy.zeros(size-1)
            for i in xrange(len(update_vector)):
                update_vector[i] += self.innerProd(self.residuals[-2], self.residuals[i])
            self.matrix[size-2,0:size-1] = update_vector
            self.matrix[0:size-1,size-2] = numpy.transpose(update_vector)

            #making the new row of the matrix 
            new_matrix = numpy.zeros((size+1,size+1))   
            new_matrix[0:size,0:size] = self.matrix     
            # upsize by one to account for the fact that the whole of the DIIS matrix 
            # is built from the error vectors in C2 DIIS 
            size = size if self.DIIS_Type == 'C1' else size + 1 
            new_vector = numpy.zeros(size)             
            for i in xrange(size):                     
                new_vector[i] = self.innerProd(self.residuals[-1], self.residuals[i])
            new_matrix[size-1,0:size] = new_vector      
            new_matrix[0:size,size-1] = numpy.transpose(new_vector)
            if self.DIIS_Type == "C1":
                new_matrix[size, 0:size] = -1.              
                new_matrix[0:size, size] = -1.             
            self.matrix = new_matrix

    def reduceSpace(self):
    # reduces the size of the fock space by one 
    # removing the oldest element
        self.matrix = numpy.delete(self.matrix,0,0)
        self.matrix = numpy.delete(self.matrix,0,1)
        self.oldAlphaFocks.pop(0) 
        self.oldBetaFocks.pop(0)
        self.alphaResiduals.pop(0) 
        self.betaResiduals.pop(0)
        self.residuals.pop(0)
    
    def makeFock(self,focks, residuals):
        new_fock = numpy.zeros(numpy.shape(focks[0]))
        size = len(self.matrix)
        DIIS_vector = numpy.append(numpy.zeros((size-1,1)),[-1.]) 
        if self.DIIS_Type == "C1":
            coeffs = numpy.linalg.solve(self.matrix,DIIS_vector)
            size -= 1      # excluding the Lagrange multiplyer 
        else:
            coeffs = self.make_C2_Coeffs(self.matrix, residuals) 
        #summing over residual matrices 
        for i in xrange(size):   
            new_fock += focks[i] * coeffs[i]
        return new_fock
    
    def make_C2_Coeffs(self, matrix, residuals):
        (_, vects) = numpy.linalg.eig(matrix)
        min_error = 1e6     # string with a arbitraily large value 
        # estimating the error associated with each possible set of coefficents 
        # and keeping track of the vector with the least associated error 
        for vector in numpy.transpose(vects):
            vector = vector / sum(vector) # normalizing the vector so the sum of its elements equals 1 
            current_error = 0
            for i in range(len(vector)):
                current_error += numpy.linalg.norm(vector[i] * residuals[i])
            if current_error < min_error:
                min_error = current_error 
                best_vect = vector 
        return(best_vect)

    def updateDIIS(self, fock, densities, overlap_matrix):
    #generates takes fock matrices and generates a new set of residuals 
    #plus monitors the size of the DIIS space
        if len(self.alphaResiduals) >= self.max_space:
            self.reduceSpace()
        self.alphaResiduals.append(self.getResidual(overlap_matrix, densities.alpha, fock.alpha))
        self.betaResiduals.append(self.getResidual(overlap_matrix, densities.beta, fock.beta))     
        self.residuals.append(self.alphaResiduals[-1] + self.betaResiduals[-1]) 
        self.oldAlphaFocks.append(fock.alpha)
        self.oldBetaFocks.append(fock.beta)
    
    # replaces the fock matrix from the previous Roothaan-Hall step
    # with the extrapolated matrix
    # How does the fact that the Fock and density matrices no longer 
    # correspond affect things?
    def storeExtrapolated(self, focks, densities, overlap_matrix):
        self.oldAlphaFocks[-1] = copy.deepcopy(focks.alpha)
        self.oldBetaFocks[-1] = copy.deepcopy(focks.beta)
        self.alphaResiduals[-1] = self.getResidual(overlap_matrix, densities.alpha, focks.alpha)
        self.betaResiduals[-1] = self.getResidual(overlap_matrix, densities.beta, focks.beta)
        self.residuals[-1] = numpy.add(self.alphaResiduals[-1], self.betaResiduals[-1])

    def DoDIIS(self, focks, densities, overlap_matrix):
        self.updateDIIS(focks, densities, overlap_matrix)
        if len(self.residuals) > 1:
            self.makeDIISMatrix()
            while numpy.linalg.cond(self.matrix) > self.max_cond:
                self.reduceSpace()
            #making the new fock matrices 
            focks.alpha = self.makeFock(self.oldAlphaFocks, self.alphaResiduals)
            focks.beta = self.makeFock(self.oldBetaFocks, self.betaResiduals)
            self.getDIISError()
            #replcing the information from the old fock matrices with that 
            #from the extrapolated matrices 
            self.storeExtrapolated(focks, densities, overlap_matrix)
        return focks 

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
    return sorted_MOs, sorted_energies, P_vector

def Sort_MOs(MOs, energies,p):
#sorts MOs and energies in decending order 
#based on a vector p (the overlap vector)
    temp = [[p[i],MOs[:,i],energies[i]] for i in range(len(p))] 
    temp = sorted(temp, key = lambda temp: temp[0], reverse = True)     #sorts the elements on the basis of the p values (temp[0])
    new_MOs = numpy.array([line[1] for line in temp])
    new_energies = [line[2] for line in temp]
    return numpy.transpose(new_MOs), new_energies

########################### Basic HF Functions ############################

class Density_matrix:
    def __init__(self, alpha_matrix, beta_matrix):
        self.alpha = alpha_matrix
        self.beta = beta_matrix
        self.total = alpha_matrix + beta_matrix 

class Fock_matrix:
    def __init__(self):
        self.core = []
        self.alpha = []
        self.beta = []

    def resetFocks(self):
    #sets the alpha and beta fock matrcies as the core 
        self.alpha = copy.deepcopy(self.core)
        self.beta = copy.deepcopy(self.core)

    def makeFockMatrices(self, densities, shell_pairs,template_matrix):
           self.resetFocks() 
           coulomb_matrix = copy.deepcopy(template_matrix)
           alpha_exchange_matrix = copy.deepcopy(template_matrix)
           beta_exchange_matrix = copy.deepcopy(template_matrix)
#           old_alpha_density_matrix = copy.deepcopy(alpha_density_matrix)
#           old_beta_density_matrix = copy.deepcopy(beta_density_matrix)
#           screen = numpy.zeroes((shell_pair1.Centre1.Cgtf.NAngMom,\
#                                 shell_pair1.Centre2.Cgtf.NAngMom,\
#                                 shell_pair2.Centre1.Cgtf.NAngMom,\
#                                 shell_pair2.Centre2.Cgtf.NAngMom)
#      Use guess/current MOs (supplied or newly calculated) to calculate two-electron contributions to Fock matrix
           for shell_pair1 in shell_pairs:
              ia_vec = shell_pair1.Centre1.Ivec
              ib_vec = shell_pair1.Centre2.Ivec
              for shell_pair2 in shell_pairs:
                 ic_vec = shell_pair2.Centre1.Ivec
                 id_vec = shell_pair2.Centre2.Ivec
#                 for m in range(0,shell_pair1.Centre1.Cgtf.NAngMom):
#                    for n in range(0,shell_pair1.Centre2.Cgtf.NAngMom):
#                       for l in range(0,shell_pair2.Centre1.Cgtf.NAngMom):
#                          for s in range(0,shell_pair2.Centre2.Cgtf.NAngMom):
#                             screen[m,n,l,s] = overlap_matrix[ia_vec[m]][ib_vec[n]]*overlap_matrix[ic_vec[l]][id_vec[s]]
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

#=================================================================#
#                                                                 #
#                        Main Function                            #
#                                                                 #
#=================================================================#

def do(system, molecule,state, alpha_reference, beta_reference):
    num_iterations = 0
    isFirstCalc = (alpha_reference[0][0] == None)
    fock = Fock_matrix() 
    if molecule.NOrbitals < system.DIIS_Size:
        system.DIIS_Size = molecule.NOrbitals
    DIIS = DIIS_system(c.DIIS_MAX_CONDITION, system.DIIS_Size, system.DIIS_Type)
    
    #setting up the values that are constant througout the calculation
    nuclear_repulsion_energy = integrals.nuclear_repulsion(molecule)
    template_matrix =  makeTemplateMatrix(molecule.NOrbitals)    
    fock.core, overlap_matrix, shell_pairs = makeCoreMatrices(template_matrix, molecule) 
    fock.resetFocks()

    s,U = numpy.linalg.eigh(overlap_matrix)
    sp = [element**-0.5e0 for element in s]
    X = numpy.dot(U,numpy.identity(len(sp))*(sp))
    Xt = numpy.transpose(X)

    #Generating the initial density matrices 
    #Note there are no reference orbitals in the first caclulation 
    if isFirstCalc == False:
        alpha_MOs, beta_MOs, density = Init.readGuess(alpha_reference, beta_reference, molecule)
    elif system.SCFGuess == 'core':
        alpha_MOs, beta_MOs, density = Init.coreGuess(fock.core, X, Xt, molecule)
    elif system.SCFGuess == 'sad':
        density = Init.sadGuess(molecule, system.BasisSets[0])
        alpha_MOs = copy.deepcopy(template_matrix)
        beta_MOs = copy.deepcopy(template_matrix)
        
    energy = calculate_energy(fock, density)
    dE = energy
    if system.UseDIIS != True:
        DIIS.error = 0     #set to zero so the convergence criterta is met when not using DIIS

    system.out.PrintInitial(nuclear_repulsion_energy, fock.core, density)
    converged = False
    #-------------------------------------------#
    #             Begin Iterations              #
    #-------------------------------------------#

    while abs(dE) > c.energy_convergence:
        num_iterations += 1 
        fock.makeFockMatrices(density, shell_pairs, template_matrix) 

       #Contrained UHF for open shell molecules
        if molecule.Multiplicity != 1:
           fock.alpha, fock.beta = constrainedUHF(overlap_matrix, density, molecule, focks)

       #preforming DIIS starting from the first variational fock matrix 
        if system.UseDIIS == True and num_iterations > 1: 
           fock = DIIS.DoDIIS(fock, density, overlap_matrix)

       #using the MOs from the previous iteration as the reference orbitals 
       #unless spcifyed in the input file
        if system.MOM_Type != "fixed":
            alpha_reference = copy.deepcopy(alpha_MOs)
            beta_reference = copy.deepcopy(beta_MOs)

        
        alpha_MOs, alpha_orbital_energies = make_MOs(X,Xt,fock.alpha)
        beta_MOs, beta_orbital_energies = make_MOs(X,Xt,fock.beta)

       #Maximum Overlap Method 
        if system.UseMOM == True and isFirstCalc == False: 
            alpha_MOs, alpha_orbital_energies, alpha_overlaps = maximumOverlapMethod(alpha_reference,
                    alpha_MOs, alpha_orbital_energies,molecule.NAlphaElectrons, overlap_matrix)
            beta_MOs, beta_orbital_energies,beta_overlaps = maximumOverlapMethod(beta_reference,
                    beta_MOs, beta_orbital_energies,molecule.NBetaElectrons, overlap_matrix)
            system.out.PrintMOM(alpha_overlaps, beta_overlaps) 

        density.alpha = make_density_matrix(density.alpha,alpha_MOs,molecule.NAlphaElectrons)
        density.beta = make_density_matrix(density.beta,beta_MOs,molecule.NBetaElectrons)
        density.total = numpy.ndarray.tolist(numpy.add(density.alpha,density.beta))            
       
        old_energy = energy
        energy = calculate_energy(fock, density)
        dE = energy - old_energy
        total_energy = energy + nuclear_repulsion_energy
   
#        if abs(dE) < c.energy_convergence and system.out.SCFPrint < system.out.SCFFinalPrint:
#            system.out.SCFPrint = system.out.SCFFinalPrint 

        system.out.PrintLoop(num_iterations, alpha_orbital_energies, beta_orbital_energies,
            density, fock, alpha_MOs, beta_MOs, dE, total_energy, DIIS.error)

        if num_iterations > 25:
            print "HF method not converging"
            break

    print molecule.Basis  
    system.out.finalPrint()
    return alpha_MOs, beta_MOs


####################################################

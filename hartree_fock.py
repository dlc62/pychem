import integrals
import Init 
import constants as c
import copy
import numpy 
import Output
from numpy import dot 
from scipy.linalg import sqrtm
numpy.set_printoptions(precision = 5, linewidth = 300)  #Makes the arrays print nicely in the output 

########################### DIIS Object ###########################

# Leave DIIS for now experiment more once the code can do larger molecules

class DIIS_System:
    def __init__(self, max_condition, system):
        self.oldFocks = []    
        self.matrix = None
        self.residuals = [] 
        self.error = 1
        self.DIIS_type = system.DIIS_Type  
        self.max_cond = max_condition
        self.max_space = system.DIIS_Size 

    def getResidual(self, overlap, density, fock, X, Xt):
        residual  = overlap.dot(density).dot(fock) - fock.dot(density).dot(overlap)
        residual = Xt.dot(residual).dot(X)
        return residual 

    def eigConditionNumber(self):
        # Finds the eigenvalue condition number of the matrix 
        u, _ = numpy.linalg.eig(self.matrix)
        u = [abs(elem) for elem in u]
        condition = max(u) / min(u)
        return condition

    def innerProd(self, matA,matB):
    #takes two matrices and returns the trace of the product 
        product = matA.dot(numpy.transpose(matB))
        return numpy.trace(product)

    def updateMatrix(self, new_matrix):
    # adds the new row to the fock matrix 
        new_vector = [self.innerProd(self.residuals[-1], res) for res in self.residuals]
        row = len(new_matrix)-1 if self.DIIS_type == 'C2' else len(new_matrix)-2 
        col = len(new_vector)
        new_matrix[:col,row] = new_vector 
        new_matrix[row,:col] = new_vector 
        return new_matrix

    def makeMatrix(self):
    # construct the DIIS matrix itself
        size = len(self.residuals) + 1
        if self.DIIS_type == 'C2':
            size -= 1 
        new_matrix = numpy.full([size,size], -1.0)
        new_matrix[-1,-1] = 0.0
        if self.matrix == None: 
            for i, res1 in enumerate(self.residuals):
                for j, res2 in enumerate(self.residuals):
                    new_matrix[i,j] = self.innerProd(res1, res2)
        else:
            new_matrix[0:size-1,0:size-1] = self.matrix
            new_matrix = self.updateMatrix(new_matrix)
        return new_matrix 

    def reduceSpace(self):
    # Removes the oldest vector from the DIIS space 
    # look at comming up with a more intelligent way of chosing which vector to remove 
        #rel_cond = numpy.linalg.cond(self.matrix) * self.matrix.max()
        condition = self.eigConditionNumber()
        while len(self.residuals) > self.max_space or condition > self.max_cond:  
            self.matrix = numpy.delete(self.matrix,0,0)
            self.matrix = numpy.delete(self.matrix,0,1)
            self.residuals.pop(0)
            self.oldFocks.pop(0)
            condition = self.eigConditionNumber()

    def getC1Coeffs(self, matrix):
        DIIS_vector = numpy.zeros([len(matrix),1])
        DIIS_vector[-1] = -1.0
        coeffs = numpy.linalg.solve(matrix, DIIS_vector)
        return coeffs[:-1]    # not returning the lagrange multiplier 

    def estimateError(self, coeffs, residuals):
        error_vect = sum([coeffs[i] * residuals[i] for i in range(len(residuals))])
        error = error_vect.max()
        return error

# C2 Currently not working 
    def getC2Coeffs(self, matrix, residuals):
        _, vects = numpy.linalg.eig(matrix)
        min_error = float("Inf")     
        best_vect = None 
        for vect in vects:
            vect /= sum(vect)         # renormalization 
            if abs(max(vect)) < 100:  # exluding vectors with large non-linearities 
                error = numpy.zeros(numpy.shape(residuals[0]))
                for i,res in enumerate(residuals):
                    error += vect[i] * res  
                error_val = numpy.linalg.norm(error)
                if error_val < min_error:
                    best_vect = vect
                    min_error = error_val
        return best_vect

    def getCoeffs(self):
        if self.DIIS_type == 'C1':
            coeffs = self.getC1Coeffs(self.matrix)
        elif self.DIIS_type == 'C2':
            coeffs = self.getC2Coeffs(self.matrix, self.residuals)
        return coeffs

    def makeFockMatrix(self, coeffs):
        new_fock = numpy.zeros(numpy.shape(self.residuals[0]))
        for i in xrange(len(coeffs)):
            new_fock += coeffs[i] * self.oldFocks[i] 
        return new_fock
   
    def updateDIIS(self, newFock, overlap, density,X, Xt):
    # includes the new extrapolated Fock vector in the DIIS space
        self.residuals[-1] = self.getResidual(overlap, density, newFock, X, Xt)
        self.oldFocks[-1] = newFock 
        new_vector = [self.innerProd(self.residuals[-1], res) for res in self.residuals]
        row = len(self.matrix)-1 if self.DIIS_type == 'C2' else len(self.matrix)-2

    def DoDIIS(self, fock, density, overlap, X, Xt):
        self.oldFocks.append(copy.deepcopy(fock))
        self.residuals.append(self.getResidual(overlap, density, fock, X, Xt))
        # start DIIS once two residuals are found 
        if len(self.residuals) > 1:
            self.matrix = self.makeMatrix()
            self.reduceSpace()
            coeffs = self.getCoeffs()
            self.error = self.estimateError(coeffs, self.residuals)
            if self.error < 0.002:
                fock = self.makeFockMatrix(coeffs) 
                self.updateDIIS(fock, overlap, density, X, Xt)
        return fock

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
       self.Index = indexi    # Index for the CGTO in atom.Basis 
       self.Ivec = index_vec  # Indexs of the angular momentum functions in a list of all angular momentum functions on the atom
    
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
          ia_vec = [(ia_count + i) for i in range(0,cgtf_a.NAngMom)]   #vector contaning the indices each angular momentum function on the atom  
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
    alphaDIIS = DIIS_System(c.DIIS_MAX_CONDITION, system)
    betaDIIS = copy.deepcopy(alphaDIIS)

    #setting up the values that are constant througout the calculation
    nuclear_repulsion_energy = integrals.nuclear_repulsion(molecule)
    template_matrix =  makeTemplateMatrix(molecule.NOrbitals)    
    fock.core, overlap_matrix, shell_pairs = makeCoreMatrices(template_matrix, molecule) 
    fock.resetFocks()
    
    # finding cannonical orthoginalization matrix
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

    system.out.PrintInitial(nuclear_repulsion_energy, fock.core, density)
    #-------------------------------------------#
    #             Begin Iterations              #
    #-------------------------------------------#
    converged = False
    while abs(dE) > c.energy_convergence:
        num_iterations += 1 
        fock.makeFockMatrices(density, shell_pairs, template_matrix) 

       #Contrained UHF for open shell molecules
        if molecule.Multiplicity != 1:
           fock.alpha, fock.beta = constrainedUHF(overlap_matrix, density, molecule, fock)

       #preforming DIIS 
        if system.UseDIIS == True and num_iterations > system.DIIS_start: 
           fock.alpha = alphaDIIS.DoDIIS(fock.alpha, density.alpha, overlap_matrix, X, Xt)
           fock.beta = betaDIIS.DoDIIS(fock.beta, density.beta, overlap_matrix, X, Xt)
        DIIS_error = max(alphaDIIS.error, betaDIIS.error)

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
            density, fock, alpha_MOs, beta_MOs, dE, total_energy, DIIS_error)

        if num_iterations > 25:
            print "HF method not converging"
            break

    system.out.finalPrint()
    return alpha_MOs, beta_MOs


####################################################

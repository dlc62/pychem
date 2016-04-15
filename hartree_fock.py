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

class DIIS_System:
    def __init__(self, max_condition, system):
        self.oldFocks = []
        self.matrix = None
        self.residuals = []
        self.error = 1
        self.DIIS_type = system.DIISType
        self.max_cond = max_condition
        self.max_space = system.DIISSize

    def getResidual(self, overlap, density, fock, X, Xt):
        residual  = overlap.dot(density).dot(fock) - fock.dot(density).dot(overlap)
        residual = Xt.dot(residual).dot(X)
        return residual

    def eigConditionNumber(self):
        # Finds the eigenvalue condition number of the matrix
        # the DIIS matrix must be a normal matrix as it is real symmetric
        u, _ = numpy.linalg.eig(self.matrix)
        u = [abs(elem) for elem in u]
        condition = max(u) / min(u)
        return condition

    def innerProd(self, matA,matB):
    #takes two matrices and returns the trace of the product
        product = matA.dot(matB.T)
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
        # Creating the fist matrix
        if self.matrix == None:
            for i, res1 in enumerate(self.residuals):
                for j, res2 in enumerate(self.residuals):
                    new_matrix[i,j] = self.innerProd(res1, res2)
        # adding a new vector to an exiting matrix
        else:
            new_matrix[0:size-1,0:size-1] = self.matrix
            new_matrix = self.updateMatrix(new_matrix)
        return new_matrix

    def reduceSpace(self,energy):
    # Removes the oldest vector from the DIIS space
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

    def getC2Coeffs(self, matrix, residuals):
        _, vects = numpy.linalg.eig(matrix)    # rewrite this to use the value calculated in finding the conition number
        min_error = float("Inf")               # Arbitrary large number
        best_vect = None
        for vect in vects:
            vect /= sum(vect)         # renormalization
            if abs(max(vect)) < 100:  # exluding vectors with large non-linearities
                error = self.estimateError(vect, residuals)
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

    def DoDIIS(self, fock, density, overlap, X, Xt, energy):
        residual = self.getResidual(overlap, density, fock, X, Xt)
        self.error = residual.max()
        threshold = -0.1 * energy
        if self.error < threshold:   # Start DIIS
            self.residuals.append(copy.deepcopy(residual))
            self.oldFocks.append(copy.deepcopy(fock))
            if len(self.residuals) > 1:
                self.matrix = self.makeMatrix()
                self.reduceSpace(energy)
                coeffs = self.getCoeffs()
                fock = self.makeFockMatrix(coeffs)
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

def Sort_MOs(MOs, energies, p):
#sorts MOs and energies in decending order
#based on a vector p (the overlap vector)
    temp = [[p[i],MOs[:,i],energies[i]] for i in range(len(p))]
    temp = sorted(temp, key = lambda pair: pair[0], reverse = True)     #sorts the elements on the basis of the p values (temp[0])
    new_MOs = numpy.array([line[1] for line in temp])
    new_energies = [line[2] for line in temp]
    return numpy.transpose(new_MOs), new_energies

def Excite(matrix, occupancy, NElectrons):
    new_matrix = copy.deepcopy(matrix)
    frm = []; to = []
    # bulding a lists of the coloumns to interchange
    for i in range(NElectrons):
        if occupancy == 0:
            frm.append(i)
    for i in range(NElectrons, len(occupancy)):
        if occupancy == 1:
            to.append(i)
    # interchanging the columns
    for i in range(len(to)):
        temp = copy.deepcopy(new_matrix[:,to[i]])
        new_matrix[:,to[i]] = new_matrix[:,frm[i]]
        new_matrix[:,frm[i]] = temp
    return new_matrix

########################### Basic HF Functions ############################

class Density_matrix:
    def __init__(self, alpha_matrix, beta_matrix):
        self.alpha = alpha_matrix
        self.beta = beta_matrix
        self.total = alpha_matrix + beta_matrix

class Fock_matrix:
    def __init__(self, n_basis_functions, direct_HF):
        self.core = []
        self.alpha = []
        self.beta = []
        # Allocating space for the two electron integrals if doing indirrect HF
        if direct_HF is False:
            self.coulomb_integrals = numpy.zeros((n_basis_functions,) * 4)      # Allocating memeory for the 4-tensors needed
            self.exchange_integrals = numpy.zeros((n_basis_functions,) * 4)     # to store the integrals

    def resetFocks(self):
    #sets the alpha and beta fock matrcies as the core
        self.alpha = copy.deepcopy(self.core)
        self.beta = copy.deepcopy(self.core)

    def makeFockMatrices(self, densities, shell_pairs, template_matrix, direct_HF, num_iterations):
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

                # Calculate integrals if direct HF or first iteration
                 if direct_HF is True or num_iterations is 1:
                    coulomb,exchange = integrals.two_electron(shell_pair1,shell_pair2)

                 for m in range(0,shell_pair1.Centre1.Cgtf.NAngMom):
                    for n in range(0,shell_pair1.Centre2.Cgtf.NAngMom):
                       for l in range(0,shell_pair2.Centre1.Cgtf.NAngMom):
                          for s in range(0,shell_pair2.Centre2.Cgtf.NAngMom):

                            # Save the integrals on the first pass of an indirect HF job
                             if direct_HF is False and num_iterations is 1:
                                self.coulomb_integrals[ia_vec[m]][ib_vec[n]][ic_vec[l]][id_vec[s]] = coulomb[m][n][l][s]
                                self.exchange_integrals[ia_vec[m]][id_vec[s]][ic_vec[l]][ib_vec[n]] = exchange[m][s][l][n]


                                #self.coulomb_integrals[ia_vec[m],ib_vec[n],ic_vec[l],id_vec[s]] = coulomb[m][n][l][s]
                                #self.exchange_integrals[ia_vec[m],id_vec[s],ic_vec[l],ib_vec[n]] = exchange[m][s][l][n]

                            # Acctually constructing the Fock matrices
                             if direct_HF is False:
                                coulomb_matrix[ia_vec[m]][ib_vec[n]] += densities.total[ic_vec[l]][id_vec[s]]* \
                                                                        self.coulomb_integrals[ia_vec[m],ib_vec[n],ic_vec[l],id_vec[s]]
                                alpha_exchange_matrix[ia_vec[m]][ib_vec[n]] += -densities.alpha[ic_vec[l]][id_vec[s]]* \
                                                                                self.exchange_integrals[ia_vec[m],id_vec[s],ic_vec[l],ib_vec[n]]
                                beta_exchange_matrix[ia_vec[m]][ib_vec[n]] += -densities.beta[ic_vec[l]][id_vec[s]]* \
                                                                               self.exchange_integrals[ia_vec[m],id_vec[s],ic_vec[l],ib_vec[n]]
                             else:
                                 coulomb_matrix[ia_vec[m]][ib_vec[n]] += densities.total[ic_vec[l]][id_vec[s]]*coulomb[m][n][l][s]
                                 alpha_exchange_matrix[ia_vec[m]][ib_vec[n]] += -densities.alpha[ic_vec[l]][id_vec[s]]*exchange[m][s][l][n]
                                 beta_exchange_matrix[ia_vec[m]][ib_vec[n]] += -densities.beta[ic_vec[l]][id_vec[s]]*exchange[m][s][l][n]

             # Form the fock matrices themselves
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
       self.Index = index     # Index for the CGTO in atom.Basis
       self.Ivec = index_vec  # Indexs of the angular momentum functions in a list of all angular momentum functions on the atom

def make_MOs(X,Xt,fock_matrix):
    transformed_fock_matrix = Xt.dot(fock_matrix).dot(X)                #orthoginalizing the fock matrix
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
    Na = molecule.Multiplicity / 2    # Dimension of active space
    Nc = molecule.NAlphaElectrons - Na         # Dimension of core space
    S = sqrtm(overlap_matrix)
    half_density_matrix = S.dot(density.total / 2).dot(S)
    NO_vals, NO_vecs = numpy.linalg.eigh(half_density_matrix)

    #Sort in order of decending occupancy
    idx = NO_vals.argsort()[::-1]           # note the [::-1] reverses the index array
    core_space = idx[:Nc]                         # Indices of the core NOs
    valence_space = idx[(Nc + Na):]               # Indices of the valence NOs
    sorted_NO_vecs = NO_vecs[:,idx]

    delta = (fock.alpha - fock.beta) / 2
    delta = delta.dot(NO_vecs)                # Transforming delta into the NO basis
    print("Delta Matrix")
    print(delta)
    lambda_matrix = numpy.zeros(numpy.shape(delta))
    for i in core_space:
        for j in valence_space:
            lambda_matrix[i,j] = -delta[i,j]
            lambda_matrix[j,i] = -delta[j,i]
    lambda_matrix = numpy.dot(lambda_matrix, NO_vects.T)  # Transforming lambda back to the AO basis

    try:
        assert numpy.allclose(lambda_matrix, lambda_matrix.T)
    except:
        print("Lambda Matrix Not Hermitian")
        import sys
        sys.exit()

    new_alpha = fock.alpha + lambda_matrix
    new_beta = fock.beta - lambda_matrix
    return new_alpha, new_beta

#=================================================================#
#                                                                 #
#                        Main Function                            #
#                                                                 #
#=================================================================#

def do(system, molecule,state, alpha_reference, beta_reference):
    num_iterations = 0
    isFirstCalc = (alpha_reference[0][0] == None)
    fock = Fock_matrix(molecule.NOrbitals, system.Direct)
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
    if isFirstCalc is False:
        alpha_reference = Excite(alpha_reference, state.AlphaOccupancy, molecule.NAlphaElectrons)
        beta_reference = Excite(beta_reference, state.BetaOccupancy, molecule.NBetaElectrons)
        alpha_MOs, beta_MOs, density = Init.readGuess(alpha_reference, beta_reference, molecule)
    elif system.SCFGuess == "READ":
        alpha_MOs, beta_MOs, density = Init.readFromFile(system.MOFileRead, molecule, template_matrix)
    elif system.SCFGuess == "CORE":
        alpha_MOs, beta_MOs, density = Init.coreGuess(fock.core, X, Xt, molecule)
    elif system.SCFGuess == "SAD":
        density = Init.sadGuess(molecule, system.BasisSets[0])
        alpha_MOs = copy.deepcopy(template_matrix)
        beta_MOs = copy.deepcopy(template_matrix)

    energy = calculate_energy(fock, density)
    dE = energy

    system.out.PrintInitial(nuclear_repulsion_energy, fock.core, density)
    #-------------------------------------------#
    #             Begin Iterations              #
    #-------------------------------------------#
    while c.energy_convergence < abs(dE):
        num_iterations += 1
        fock.makeFockMatrices(density, shell_pairs, template_matrix, system.Direct, num_iterations)

        if system.Reference == "CUHF":
            fock.alpha, fock.beta = constrainedUHF(overlap_matrix, density, molecule, fock)

       #performing DIIS
        if system.UseDIIS == True:
           fock.alpha = alphaDIIS.DoDIIS(fock.alpha, density.alpha, overlap_matrix, X, Xt, energy)
           fock.beta = betaDIIS.DoDIIS(fock.beta, density.beta, overlap_matrix, X, Xt, energy)
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
            #system.out.PrintMOM(alpha_overlaps, beta_overlaps)

        density.alpha = make_density_matrix(density.alpha,alpha_MOs,molecule.NAlphaElectrons)
        density.beta = make_density_matrix(density.beta,beta_MOs,molecule.NBetaElectrons)
        density.total = numpy.add(density.alpha,density.beta)

        old_energy = energy
        energy = calculate_energy(fock, density)
        dE = energy - old_energy
        total_energy = energy + nuclear_repulsion_energy

#        if abs(dE) < c.energy_convergence and system.out.SCFPrint < system.out.SCFFinalPrint:
#            system.out.SCFPrint = system.out.SCFFinalPrint

        system.out.PrintLoop(num_iterations, alpha_orbital_energies, beta_orbital_energies,
            density, fock, alpha_MOs, beta_MOs, dE, total_energy, DIIS_error)

        if num_iterations >= system.MaxIterations:

            print("SCF not converging")
            break
    system.out.finalPrint()

    return alpha_MOs, beta_MOs

####################################################

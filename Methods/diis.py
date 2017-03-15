# System libraries
import numpy
from numpy import dot
import scipy
from scipy.linalg import sqrtm
import copy

# Custom-written data modules
import Data.constants as c

# Custom-written code modules
from Util import util

# The argument 'error_vec' determins which of the two possible residual vectors are used
# error_vec = 'commute' uses the commutator of the fock and density matrices - this works best for ground states
# error_vec = 'diff' uses the difference between sucessive fock matrices - this works best for excited states

#======================================================================#
#                           MAIN ROUTINES                              #
#======================================================================#
#----------------------------------------------------------------------#
#                           Entry Point                                #
#----------------------------------------------------------------------#

def do(molecule, this, settings, error_vec):

    # Set up error estimates
    if error_vec == "commute":
        alpha_residual = get_residual_com(molecule.Overlap, this.Alpha.Density, this.Alpha.Fock, molecule.Xt, molecule.X)
        beta_residual = get_residual_com(molecule.Overlap, this.Beta.Density, this.Beta.Fock, molecule.Xt, molecule.X)
    else:
        alpha_residual = get_residual_diff(this.AlphaDIIS.pre_DIIS_fock, this.Alpha.Fock)
        beta_residual = get_residual_diff(this.BetaDIIS.pre_DIIS_fock, this.Beta.Fock)

    this.AlphaDIIS.Error = alpha_residual.max()
    this.BetaDIIS.Error = beta_residual.max()
    settings.DIIS.Threshold = abs(0.2 * this.Energy)

    # Get Coordinates
    alpha_coeffs = get_coeffs(alpha_residual, this.Alpha.Fock, this.Alpha.Density, this.AlphaDIIS, settings, molecule)
    beta_coeffs = get_coeffs(beta_residual, this.Beta.Fock, this.Beta.Density, this.BetaDIIS, settings, molecule)

    # update the matrices
    if alpha_coeffs[0] != None and max(alpha_coeffs) < 3:
        this.Alpha.Fock = extrapolate_matrices(this.AlphaDIIS.OldFocks, alpha_coeffs)
        this.Alpha.Density = extrapolate_matrices(this.AlphaDIIS.OldDensities, alpha_coeffs)
    if beta_coeffs[0] != None and max(beta_coeffs) < 3:
        this.Beta.Fock = extrapolate_matrices(this.BetaDIIS.OldFocks, beta_coeffs)
        this.Beta.Density = extrapolate_matrices(this.BetaDIIS.OldDensities, beta_coeffs)
    this.Total.Density = this.Alpha.Density + this.Beta.Density

#----------------------------------------------------------------------#
#                           DIIS Procedure                             #
#----------------------------------------------------------------------#

def get_coeffs(residual, fock, density, DIIS, settings, molecule):
    DIIS.pre_DIIS_fock = fock
    under_thresh = abs(residual).max() < settings.DIIS.Threshold  # Check the matrix is good enoght to start DIIS
    non_zero = residual.any()                                     # check the residual is non_zero
    if under_thresh and non_zero:
        DIIS.Residuals.append(residual)
        DIIS.OldFocks.append(fock)
        DIIS.OldDensities.append(density)
        if len(DIIS.Residuals) > 1:
            make_diis_matrix(DIIS,settings)
            reduce_space(DIIS,settings)
            coeffs = solve_coeffs(DIIS,settings)
    return [None]    # Return None to indicate residual was too large

#======================================================================#
#                             SUBROUTINES                              #
#======================================================================#
#----------------------------------------------------------------------#

def get_residual_com(overlap, density, fock, X, Xt):
    residual  = overlap.dot(density).dot(fock) - fock.dot(density).dot(overlap)
    residual = Xt.dot(residual).dot(X)
    return residual

def get_residual_diff(old_fock, new_fock):
    if old_fock[0][0] == None:
        return new_fock
    else:
        return new_fock - old_fock

#----------------------------------------------------------------------#

def make_diis_matrix(DIIS, settings):
    size = len(DIIS.Residuals) + 1
    if settings.DIIS.Type == 'C2':
        size -= 1
#    new_matrix = numpy.full([size,size], -1.0)
    new_matrix = numpy.zeros((size,) * 2)
    new_matrix.fill(-1.0)
    new_matrix[-1,-1] = 0.0
    # Create the first matrix
    if DIIS.Matrix[0][0] == None:
        for i, res1 in enumerate(DIIS.Residuals):
            for j, res2 in enumerate(DIIS.Residuals):
                new_matrix[i,j] = util.inner_product(res1, res2)
    # Add a new vector to an existing matrix
    else:
        new_matrix[0:size-1,0:size-1] = DIIS.Matrix
    DIIS.Matrix = update_diis_matrix(DIIS, new_matrix, settings)

def update_diis_matrix(DIIS, new_matrix, settings):
    # adds the new row to the fock matrix
    new_vector = [util.inner_product(DIIS.Residuals[-1], res) for res in DIIS.Residuals]
    row = len(new_matrix)-1 if settings.DIIS.Type == 'C2' else len(new_matrix)-2
    col = len(new_vector)
    new_matrix[:col,row] = new_vector
    new_matrix[row,:col] = new_vector
    return new_matrix

#----------------------------------------------------------------------#

def reduce_space(DIIS, settings):
# Removes the oldest vector from the DIIS space
    condition = util.eigenvalue_condition_number(DIIS.Matrix)
    while len(DIIS.Residuals) > settings.DIIS.Size or condition > settings.DIIS.MaxCondition:
        DIIS.Matrix = numpy.delete(DIIS.Matrix,0,0)
        DIIS.Matrix = numpy.delete(DIIS.Matrix,0,1)
        DIIS.Residuals.pop(0)
        DIIS.OldFocks.pop(0)
        DIIS.OldDensities.pop(0)
        condition = util.eigenvalue_condition_number(DIIS.Matrix)

#----------------------------------------------------------------------#

def solve_coeffs(DIIS, settings):
    matrix = damp_matrix(DIIS.Matrix, settings.DIIS.Damping)
    if settings.DIIS.Type == 'C1':
        coeffs = get_C1_coeffs(matrix)
    elif settings.DIIS.Type == 'C2':
        coeffs = get_C2_coeffs(matrix, DIIS.Residuals)
    return coeffs

def damp_matrix(matrix, damping):
    damped_matrix = copy.deepcopy(matrix)
    damped_matrix[numpy.diag_indices(len(damped_matrix))] *= (1.0 + damping)
    return damped_matrix

def get_C1_coeffs(matrix):
    DIIS_vector = numpy.zeros([len(matrix),1])
    DIIS_vector[-1] = -1.0
    coeffs = numpy.linalg.solve(matrix, DIIS_vector)
    return coeffs[:-1][:,0]    # not returning the lagrange multiplier

def get_C2_coeffs(matrix, residuals):
    eigvals, vects = numpy.linalg.eigh(matrix)
    min_error = float("Inf")               # Arbitrary large number
    best_vect = None
    for vect in vects:
        vect /= sum(vect)         # renormalization
        if abs(max(vect)) < 10:   # exluding vectors with large non-linearities
            error = estimate_error(vect, residuals)
            error_val = numpy.linalg.norm(error)
            if error_val < min_error:
                best_vect = vect
                min_error = error_val
    return best_vect

def estimate_error(coeffs, residuals):
    error_vect = sum([coeffs[i] * residuals[i] for i in range(len(residuals))])
    error = error_vect.max()
    return error

#----------------------------------------------------------------------#

def extrapolate_matrices(matrices, coeffs):
    new_matrix = numpy.zeros(numpy.shape(matrices[0]))
    for i in range(len(coeffs)):
        new_matrix += coeffs[i] * matrices[i]
    return new_matrix

#----------------------------------------------------------------------#

def reset_diis(DIIS):
    DIIS.Residuals = []
    DIIS.Matrix = [[None]]
    DIIS.OldFocks = []
    DIIS.OldDensities = []

def check_coeffs(coeffs):
    if len(coeffs) is 1:
        return False
    else:
        return coeffs[-2] < 1.1 and coeffs[-2] > 0.9

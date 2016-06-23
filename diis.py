# System libraries
import numpy
from numpy import dot
import scipy
from scipy.linalg import sqrtm
import copy

# Custom-written data modules
import constants as c

# Custom-written code modules
import util

#======================================================================#
#                           MAIN ROUTINES                              #
#======================================================================#
#----------------------------------------------------------------------#
#                           Entry Point                                #
#----------------------------------------------------------------------#

def do(molecule, this, settings):

    # Set up error estimates
    alpha_residual = get_residual(molecule.Overlap, this.Alpha.Density, this.Alpha.Fock, molecule.X, molecule.Xt)
    beta_residual = get_residual(molecule.Overlap, this.Beta.Density, this.Beta.Fock, molecule.X, molecule.Xt)
    this.AlphaDIIS.Error = alpha_residual.max()
    this.BetaDIIS.Error = beta_residual.max()
    settings.DIIS.Threshold = -0.1 * this.Energy

    # Perform DIIS procedure
    this.Alpha.Fock = diis(alpha_residual, this.Alpha.Fock, this.AlphaDIIS, settings)
    this.Beta.Fock = diis(beta_residual, this.Beta.Fock, this.BetaDIIS, settings)

#----------------------------------------------------------------------#
#                           DIIS Procedure                             #
#----------------------------------------------------------------------#

def diis(residual, fock, DIIS, settings):
    if residual.max() < settings.DIIS.Threshold:
        DIIS.Residuals.append(residual)
        DIIS.OldFocks.append(fock)
        if len(DIIS.Residuals) > 1:
            make_diis_matrix(DIIS,settings)
            reduce_space(DIIS,settings)
            coeffs = get_coeffs(DIIS,settings)
            fock = make_fock_matrix(DIIS,coeffs)
    return fock

#======================================================================#
#                             SUBROUTINES                              #
#======================================================================#
#----------------------------------------------------------------------#

def get_residual(overlap, density, fock, X, Xt):
    residual  = overlap.dot(density).dot(fock) - fock.dot(density).dot(overlap)
    residual = Xt.dot(residual).dot(X)
    return residual

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
    if DIIS.Matrix == None:
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
        condition = util.eigenvalue_condition_number(DIIS.Matrix)

#----------------------------------------------------------------------#

def get_coeffs(DIIS, settings):
    if settings.DIIS.Type == 'C1':
        coeffs = get_C1_coeffs(DIIS.Matrix)
    elif settings.DIIS.Type == 'C2':
        coeffs = get_C2_coeffs(DIIS.Matrix, DIIS.Residuals)
    return coeffs

def get_C1_coeffs(matrix):
    DIIS_vector = numpy.zeros([len(matrix),1])
    DIIS_vector[-1] = -1.0
    coeffs = numpy.linalg.solve(matrix, DIIS_vector)
    return coeffs[:-1]    # not returning the lagrange multiplier

def get_C2_coeffs(matrix, residuals):
    eigvals, vects = numpy.linalg.eig(matrix)
    min_error = float("Inf")               # Arbitrary large number
    best_vect = None
    for vect in vects:
        vect /= sum(vect)         # renormalization
        if abs(max(vect)) < 100:  # exluding vectors with large non-linearities
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

def make_fock_matrix(DIIS, coeffs):
    new_fock = numpy.zeros(numpy.shape(DIIS.Residuals[0]))
    for i in range(len(coeffs)):
        new_fock += coeffs[i] * DIIS.OldFocks[i]
    return new_fock

#----------------------------------------------------------------------#

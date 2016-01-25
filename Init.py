#This version supports the OO hartree_fock.py

import numpy
from copy import deepcopy
import hartree_fock as HF
from SAD_orbitals import SADget


def dirrectSum(A,B):
    if A == []:
        new_matrix = numpy.array(B)
    else:
        size = len(A) + len(B)
        new_matrix = numpy.zeros((size,size))
        new_matrix[:len(A), :len(A)] = A
        new_matrix[len(B):, len(B):] = B
    return new_matrix

def sadGuess(molecule,basis):
    density_matrix = []
    for atom in molecule.Atoms:
        density_matrix = dirrectSum(density_matrix, SADget[basis][atom.Label]) 
    density_matrix *= 0.5
    return HF.Density_matrix(density_matrix, deepcopy(density_matrix)) 

def readGuess(alpha_ref, beta_ref, state, molecule):
    alpha_density = HF.makeTemplateMatrix(molecule.NOrbitals) 
    beta_density = HF.makeTemplateMatrix(molecule.NOrbitals) 
    alpha_MOs = HF.Excite(alpha_ref, state.AlphaOccupancy, molecule.NAlphaElectrons)
    beta_MOs = HF.Excite(beta_ref, state.BetaOccupancy, molecule.NBetaElectrons)
    alpha_density = HF.make_density_matrix(alpha_density, alpha_MOs, molecule.NAlphaElectrons)
    beta_density = HF.make_density_matrix(beta_density, beta_MOs, molecule.NBetaElectrons)
    return alpha_MOs, beta_MOs, HF.Density_matrix(alpha_density, beta_density) 

def coreGuess(core_fock, X, Xt, molecule):
    MOs, energies = HF.make_MOs(X, Xt, core_fock)
    template_matrix = HF.makeTemplateMatrix(molecule.NOrbitals)
    alpha_density = HF.make_density_matrix(template_matrix, MOs, molecule.NAlphaElectrons)
    beta_density = deepcopy(alpha_density)
    return MOs, deepcopy(MOs), HF.Density_matrix(alpha_density, beta_density) 

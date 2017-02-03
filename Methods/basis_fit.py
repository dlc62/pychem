# System libraries
import numpy
import copy
from scipy.misc import factorial2
# Custom-written data modules
from Data import basis
from Data.constants import lQuanta, nAngMomFunctions

def do(molecule, MOs, new_basis):

    #iterating over each MO in the old basis
    for MO in range(len(MOs)):
        if MO is not 0 and MO is size:
            break
        #old_coeffs = numpy.ndarray.tolist(MOs[:,MO])      # pulling out the MO coefficents associated with a single MO
        old_coeffs = MOs[:,MO]
        new_MO = numpy.array([])                                        # single column in the eventual MO matrix
        cgto_count = 0
        for atom in molecule.Atoms:
            coeffs = Basis_Fit_Atom(atom, old_coeffs, cgto_count, new_basis, molecule.Basis)
            new_MO = numpy.append(new_MO, coeffs)
            cgto_count += atom.NFunctions                 # keeps track of the index to fit next
        #initializing matrix to store the coeffs once its size is known
        if MO == 0:
            size = len(new_MO)
            new_coeffs = numpy.zeros((size, size))
        new_coeffs[:,MO] = new_MO
    return new_coeffs

#---------------------------------------------------------------------------------------

def Get_Overlap(prim1, prim2, l, m=0):
    #gamma = prim1[0] + prim2[0]
    #norm = (prim1[0]*prim2[0])**(3./4 + l/2.)
    #integral = (1 / gamma)**(3./2) * (1/(2*gamma)**l) * norm * prim1[1] * prim2[1]
    gamma = prim1[0] + prim2[0]
    factorials = numpy.product(factorial2(2 * numpy.array(lQuanta[l][m]) - 1))
    integral = prim1[1] * prim2[1] / (2*gamma) ** l * (3.142 / gamma) ** (3./2) * factorials
    return integral

#---------------------------------------------------------------------------------------

def Basis_Fit_Atom(atom, MOs, cgto_count, new_basis, old_basis):
    new_cgtos = basis.get[new_basis][atom.Label]
    old_ang_indices = get_ang_indices(atom, basis.get[old_basis][atom.Label], cgto_count)
    new_ang_indices = get_ang_indices(atom, new_cgtos)
    size = sum([len(l) for l in new_ang_indices])                 # getting the number of cgtos centered on this atom in the new basis
    atom_coeffs = numpy.zeros(size)
    for Ang in range(atom.MaxAng+1):    #iterating over angular momentum quantum numbers
        degen = nAngMomFunctions[Ang]
        old_idx = old_ang_indices[Ang]
        new_idx = new_ang_indices[Ang]
        ang_set = [cgto for cgto in atom.Basis if cgto.AngularMomentum is Ang]
        #Getting the list of new functions of the correct angular momentum
        NewFunctions = [cgto[1:] for cgto in new_cgtos if cgto[0] is Ang]
        if ang_set != []:
            coeffs = [0.0] * len(NewFunctions) * degen
            for m in range(degen):                                  # iterating over magnetic quantum numbers
                m_coeffs = Basis_Fit_Ang(atom, ang_set, MOs[old_idx], NewFunctions, m)
                for i in range(len(m_coeffs)):
                    coeffs[i*degen + m] = m_coeffs[i]
        atom_coeffs[new_idx] = coeffs                               # building up the vec of coeffs
        cgto_count += len(ang_set) * degen

    return atom_coeffs

#---------------------------------------------------------------------------------------

def Basis_Fit_Ang(atom, old_set, MOs, new_ang_set, m):    #Take all the MO coefficents for the state
    #Getting the set of primitive functions for the old basis
    cgto_count = m
    ang_set = [cgto.Primitives for cgto in old_set]
    Ang = old_set[0].AngularMomentum

    funcs = range(len(new_ang_set))                               #List of indices for the new basis functions
    #Finding the overlap matrix for the new orbitals
    S = numpy.zeros((len(funcs), len(funcs)))
    for orb1 in funcs:
        for orb2 in funcs[0:orb1+1]:
            for prim1 in new_ang_set[orb1]:
                for prim2 in new_ang_set[orb2]:
                    S[orb1][orb2] += Get_Overlap(prim1, prim2, Ang, m)
    S += numpy.transpose(numpy.tril(S,-1))

    #Extracting the old primitives and multiplying them by the MO coefficents
    old_set = copy.deepcopy(ang_set)
    for cgto in range(len(old_set)):
        for prim in old_set[cgto]:
            prim[1] *=  MOs[cgto_count]
        cgto_count += (2*Ang + 1)            # This is assuming the MOs are ordered by l

    #Finding the overlap of the new functions with the old
    T = numpy.array([0.0 for i in funcs])
    for orb1 in funcs:                       #iterating over new functions
        for orb2 in range(len(old_set)):     #iterating over old functions
            for prim1 in new_ang_set[orb1]:
                for prim2 in old_set[orb2]:
                    T[orb1] += Get_Overlap(prim1,prim2,Ang, m)

    new_MOs = numpy.linalg.solve(S,T)
    return numpy.ndarray.tolist(new_MOs)

def get_ang_indices(atom, cgtos, start=0):
    """ Gets the indices of the MO coefficents associated with each l value
        for the atom """
    indices = [[] for l in range(atom.MaxAng+1)]
    index_count = start
    for cgto in cgtos:
        ang = cgto[0]
        degen = nAngMomFunctions[ang]
        indices[ang] += range(index_count, index_count+degen)
        index_count += degen
    return indices

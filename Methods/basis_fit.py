import numpy 
import copy
# Import custom-written data modules
from Data import basis

def do(molecule, MOs, new_basis):
    #iterating over each MO in the old basis 
    for MO in xrange(len(MOs)):
        old_coeffs = numpy.ndarray.tolist(MOs[:,MO])      # pulling out the MO coefficents associated with a single state 
        new_MO = []                                       # single column in the eventual MO matrix
        cgto_count = 0 
        for atom in molecule.Atoms:
            coeffs = Basis_Fit_Atom(atom, old_coeffs, cgto_count, new_basis)    
            new_MO += coeffs
            cgto_count += atom.NFunctions                 # keeps track of the index to fit next 
        #initializing matrix to store the coeffs once its size is known
        if MO == 0:
            size = len(new_MO)
            new_coeffs = [[0.0]*size]*size
        new_coeffs[MO] = new_MO
    return numpy.transpose(numpy.array(new_coeffs))    
    
#---------------------------------------------------------------------------------------
    
def Get_Overlap(prim1, prim2, l):
    gamma = prim1[0] + prim2[0]
#    fact = factorial2(2*l-1,exact = True)
#    norm = ((2 ** l) * (prim1[0]*prim2[0])**(3./4 + l/2.)) / (pi**3./2 * fact)
#    integral = (pi / gamma)**(3./2) * (fact/(2*gamma)**l) * norm * prim1[1] * prim2[1]
    norm = (prim1[0]*prim2[0])**(3./4 + l/2.)
    integral = (1 / gamma)**(3./2) * (1/(2*gamma)**l) * norm * prim1[1] * prim2[1]
    return integral 
    
#---------------------------------------------------------------------------------------

def Basis_Fit_Atom(atom, MOs, cgto_count, new_basis):
    atom_coeffs = []
    for Ang in xrange(atom.MaxAng+1):    #iterating over angular momentum quantum numbers
        degen = 2*Ang + 1 
        ang_set = []
        for cgto in atom.Basis:
            if cgto.AngularMomentum == Ang:
                ang_set.append(cgto)
        #Getting the list of new functions of the correct angular momentum  
        NewFunctions = []
        for cgto in basis.get[new_basis][atom.Label]:
            if cgto[0] == Ang:
                NewFunctions.append(cgto[1:])
        if ang_set != []:
            coeffs = [0.0] * len(NewFunctions) * degen
            for m in range(degen):                          #iterating over magnetic quantum numbers 
                m_coeffs = Basis_Fit_Ang(atom, ang_set, MOs, cgto_count+m, NewFunctions)
                for i in range(len(m_coeffs)):
                    coeffs[i*degen + m] = m_coeffs[i]
        atom_coeffs += coeffs
        cgto_count += len(ang_set) * degen
        
    #Adding on coefficents of 0 for polarizing functions in the new basis 
    newMaxAng = max([cgto[0] for cgto in basis.get[new_basis][atom.Label]])
    if newMaxAng > atom.MaxAng:
        for cgto in basis.get[new_basis][atom.Label]:
            if cgto[0] > atom.MaxAng:
                atom_coeffs += [0.0] *  (2 * cgto[0] + 1)
                cgto_count += 2 * cgto[0] + 1 
                    
    return atom_coeffs 
       
#---------------------------------------------------------------------------------------

def Basis_Fit_Ang(atom, old_set, MOs, cgto_count, new_ang_set):    #Take all the MO coefficents for the state
    
    #Getting the set of function of the right l from the new basis 
    ang_set = [cgto.Primitives for cgto in old_set]
    Ang = old_set[0].AngularMomentum
    
    funcs = range(len(new_ang_set))                               #List of indices for the new basis functions 
    #Finding the overlap matrix for the new orbitals 
    S = numpy.array([[0.0 for i in funcs] for j in funcs])
    for orb1 in funcs:
        for orb2 in funcs[0:orb1+1]:
            for prim1 in new_ang_set[orb1]:
                for prim2 in new_ang_set[orb2]:
                    #norm = (4 * (prim1[0] * prim2[0]) / pi**2)**(3./4)
                    #S[orb1][orb2] += prim1[1] * prim2[1] * (pi/(prim1[0] + prim2[0]))**(3./2) * norm
                    S[orb1][orb2] += Get_Overlap(prim1, prim2, Ang)
    S += numpy.transpose(numpy.tril(S,-1))

    #Extracting the old primitives and multiplying them by the MO coefficents
    old_set = copy.deepcopy(ang_set)
    for cgto in range(len(old_set)):
        for prim in old_set[cgto]:
            prim[1] *= MOs[cgto_count]
        cgto_count += (2*Ang + 1)
                
    #Finding the overlap of the new functions with the old
    T = numpy.array([0.0 for i in funcs])
    for orb1 in funcs:                       #iterating over new functions 
        for orb2 in range(len(old_set)):     #iterating over old functions 
            for prim1 in new_ang_set[orb1]:
                for prim2 in old_set[orb2]:
                    #norm = (4 * (prim1[0] * prim2[0]) / pi**2)**(3./4)
                    #T[orb1] += prim1[1] * prim2[1] * (pi/(prim1[0] + prim2[0]))**(3./2) * norm
                    T[orb1] += Get_Overlap(prim1,prim2,Ang)

    new_MOs = numpy.linalg.solve(S,T)
    return numpy.ndarray.tolist(new_MOs)     
    

#!/usr/bin/python

import sys
import input
import basis
import Output
import constants as c
import hartree_fock
import Basis_Fitting
from copy import deepcopy
import numpy
import cProfile


def single_excitations(n_electrons, n_orbitals):
    """Takes the number of electrons of a particular spin and the number
    of orbitals and returns the list of pairs corresponding to all single
    excitations"""
    excitations = []
    n_virtual_orbitals = n_orbitals - n_electrons
    for i in range(1,n_electrons+1):
        for j in range(1,n_virtual_orbitals+1):
            excitations.append([-i,j])
    return excitations

def make_length_equal(list1, list2, place_holder = []):
    diff = len(list1) - len(list2)
    extra = [place_holder for i in range(abs(diff))]
    if diff > 0:
        list2 += extra
    elif diff < 0:
        list1 += extra


#----------------------------------------------------------------
# Set up data structures as python classes
# Convention: class variables are capitalized, instances not
#----------------------------------------------------------------
# System specifies the values for a single calculation
# This will be written so that multiple system instances can be produced
# allowing for more flexible series os of calculations

class System:
    def __init__(self,input):
        try:
           self.Method = input.Method.upper()
        except:
           self.Method = 'HF'
        try:
           self.JobType = input.JobType.upper()
        except:
           self.JobType = 'ENERGY'
        try:
           self.BasisSets = input.BasisSets
        except:
           print 'Error: must provide BasisSets list'
           sys.exit()
        try:
           assert type(input.BasisSets) is list
        except AssertionError:
           print 'Error: basis sets must be specified as strings in a list'
           sys.exit()
        try:
           assert len(input.BasisSets) > 0
        except AssertionError:
           print 'Error: must specify at least one basis set'
           sys.exit()
        try:
           self.BasisFit = input.BasisFit
        except:
           if len(input.BasisSets) == 1:
              self.BasisFit = False
           else:
              self.BasisFit = True
        try:
            self.MaxIterations = input.MaxIterations
        except:
            self.MaxIterations = 25
        try:
            self.Direct = input.Direct
        except:
            self.Direct = True
        try:
            self.Reference = input.Reference.upper()
        except:
            self.Reference = "UHF"
############ DIIS Settings #############
        if self.Reference == "CUHF":
            self.UseDIIS = False
        else:
            try:
                self.UseDIIS = input.UseDIIS
            except:
                self.UseDIIS = True
        try:
            self.DIISSize = input.DIIS_Size
        except:
            self.DIISSize = 15
        try:
            self.DIISType = input.DIIS_Type
            if self.DIIS_Type not in ["C2", "C1"]:
                print("DIIS type must be C2 or C1 using C1 by default")
                sys.exit()
        except:
            self.DIISType = "C1"
        try:
            self.DIISstart = input.DIIS_Start
        except:
            self.DIISstart = 1

############ Initial Guess Settings ##########
        try:
            #will need to add code to check if the data is avalible for SAD guess
            # for the givej molecule and basis
            self.SCFGuess = input.SCFGuess.upper()
            assert(self.SCFGuess in ["READ", "CORE", "SAD"])
        except:
            print("Could not read SCF guess, defualting to core guess")
            self.SCFGuess = "CORE"
        if self.SCFGuess == "READ":
            try:
                self.MOFileRead = input.MOFileRead
            except:
                print("Specify a file for the input MOs")
                sys.exit()

############ MOM Settings
        try:
            self.UseMOM = input.UseMOM
        except:
            try:
                input.Excitations
                self.UseMOM = True
            except:
                try:
                    input.Alpha_Excitations
                    self.UseMOM = True
                except:
                    self.UseMOM = False
        try:
            # at this point the code just defaults to using the previous orbials as
            # the reference at each iteration, will need to change the initialzation
            # to check for the 'fixed' keyword to used fixed reference orbitals
            self.MOM_Type = input.MOM_Type.upper()
        except:
            self.MOM_Type = "MUTABLE"
        self.out = Output.PrintSettings()

#---------------------------------------------------#
#                  Molecule Class                   #
#---------------------------------------------------#

# Possibly edit molecule class
#    - Make it easy to get coords out
#    - Store a list of basis functions

class Molecule:

    def do_excitation(self, n_electrons, ground_occ, excitation):
        occupyed = deepcopy(ground_occ)
        if excitation != []:
            occupyed[n_electrons+excitation[0]] = 0
            occupyed[n_electrons + excitation[1] - 1] = 1
        return occupyed

    def make_excitations(self, ground, alpha_excitations, beta_excitations):
        self.States = [ground]
        alpha_ground = ground.AlphaOccupancy
        beta_ground = ground.BetaOccupancy
        num_orbitals = len(alpha_ground)


        if alpha_excitations == beta_excitations == [[]]:
            # immeditatly returns if no excitations were specifyed
            return self.States
        elif alpha_excitations == 'Single':
            alpha_excitations = single_excitations(self.NAlphaElectrons, num_orbitals)
        elif alpha_excitations == 'Double':
            alpha_singles = single_excitations(self.NAlphaElectrons, num_orbtials)
            beta_singles = single_excitations(self.NBetaElectrons, num_orbitals)
            for excite1 in alpha_singles:
                for excite2 in beta_singles:
                    pass
        # else manual specification of orbitals

        make_length_equal(alpha_excitations, beta_excitations)
        # Common loop to make the excitations for all cases
        for i in range(len(alpha_excitations)):
            try:
                assert type(alpha_excitations[i]) is list
                assert type(beta_excitations[i]) is list
            except AssertionError:
                print("Each excitation must be a pair of two integers")
                sys.exit()
            alpha_occupied = self.do_excitation(self.NAlphaElectrons, alpha_ground, alpha_excitations[i])
            beta_occupied = self.do_excitation(self.NBetaElectrons, beta_ground, beta_excitations[i])
            self.States += [(ElectronicState(alpha_occupied, beta_occupied))]
        return self.States

    def __init__(self,input,coords,basis_set):
        self.Basis = basis_set
        try:
           self.Charge = input.Charge
        except:
           print 'Error: must specify molecule charge using Charge ='
           sys.exit()
        try:
           self.Multiplicity = input.Multiplicity
        except:
           print 'Error: must specify molecule multiplicity using Multiplicity ='
           sys.exit()
        try:
           assert type(coords) is list
           assert type(coords[0]) is list
           assert len(coords[0]) == 5
        except AssertionError:
           print 'Coordinates must be specified in list of lists format'
           print 'Each atom entry is of form [atom_symbol,atom_nuc_charge,x,y,z]'
           sys.exit()
#       Set up atoms within molecule, calculate number of electrons and ground state electronic configuration
        self.NAtom = len(coords)
        #Initializing everything to 0
        self.Atoms = []
        n_electrons = 0
        n_orbitals = 0
        n_core_orbitals = 0
        index = 0
        for row in coords:
            atom = Atom(index,row,basis_set)
            self.Atoms.append(atom)
            n_electrons += c.nElectrons[atom.Label]
            n_core_orbitals += c.nCoreOrbitals[atom.Label]
            n_orbitals += atom.NFunctions
            index += 1
        n_electrons = n_electrons - self.Charge
        try:
            n_alpha = (n_electrons + (self.Multiplicity-1))/2
            n_beta  = (n_electrons - (self.Multiplicity-1))/2
        except:
            print 'Error: charge and multiplicity inconsistent with specified molecule'
            sys.exit()
        self.NElectrons = n_electrons
        self.NAlphaElectrons = n_alpha
        self.NBetaElectrons = n_beta
        self.NOrbitals = n_orbitals
#        n_valence_orbitals = n_orbitals - n_core_orbitals
        alpha_occupied = [1 for i in range(0,n_alpha)]
        beta_occupied = [1 for i in range(0,n_beta)]
        alpha_unoccupied = [0 for i in range(0,n_orbitals-n_alpha)]
        beta_unoccupied = [0 for i in range(0,n_orbitals-n_beta)]
#       combining the occupyed and unoccupyed lists to make two total occupancy lists
        alpha_occupancy = alpha_occupied + alpha_unoccupied    ####  These lines have been changed ####
        beta_occupancy = beta_occupied + beta_unoccupied
#       Set up alpha and beta occupancy lists describing different electronic states, including ground state first

###################  Excitations ####################

        alpha_excitations = [[]]
        beta_excitations = [[]]
        try:
            alpha_excitations = input.Alpha_Excitations
            beta_excitations = input.Beta_Excitations
        except AttributeError:
            try:
                alpha_excitations = input.Excitations
            except:
                pass
        # Cecking that excitations were given in the correct format
        try:
            alpha_is_list = type(alpha_excitations) is list
            alpha_is_keyword = alpha_excitations in ['Single', 'Double']
            assert alpha_is_list or alpha_is_keyword
            assert type(beta_excitations) is list
        except AssertionError:
            print("""Excitations must be specified as a list of excitations each containing two elements
            or using the keywords 'Single' or 'Double'""")
            sys.exit()
        ground = ElectronicState(alpha_occupancy,beta_occupancy)
        self.States = self.make_excitations(ground, alpha_excitations, beta_excitations)

################# End Excitations ##################

class Atom:
    def __init__(self,index,row,basis_set):
        [label,Z,x,y,z] = row
        self.Index = index
        self.Label = label.upper()
        self.NuclearCharge = Z
        self.Coordinates = [x*c.toBohr,y*c.toBohr,z*c.toBohr]
        self.Basis = []
        self.NFunctions = 0
        self.MaxAng = 0
        basis_data = basis.get[basis_set][self.Label]
        for function in basis_data:
            self.Basis.append(ContractedGaussian(function))
            self.NFunctions += 2*function[0] + 1
            if function[0] > self.MaxAng:
                self.MaxAng = function[0]
    def update_coords(xyz):
        self.Coordinates = xyz

class ContractedGaussian:
    def __init__(self,function):
        self.AngularMomentum = function[0]
        self.NAngMom = c.nAngMomFunctions[self.AngularMomentum]
        self.Primitives = function[1:]
        self.NPrimitives = len(self.Primitives)
        self.NFunctions = self.AngularMomentum * self.NAngMom

class ElectronicState:
    def __init__(self,alpha_occupancy,beta_occupancy):
        #Initializer takes the ground state occupancies and then constrcts the new
        #occupancy using these and the excitations
        self.AlphaOccupancy = alpha_occupancy
        self.BetaOccupancy = beta_occupancy
    def update(self,energy,gradient,hessian,MOs):
        self.Energy = energy
        self.Gradient = gradient
        self.Hessian = hessian
        self.MolecularOrbitals = MOs

#-------------- Utility Functions --------------#

def remove_punctuation(basis_set):
    basis_set = basis_set.replace('*','s').replace('-','').replace('(','').replace(')','').replace(',','').upper()
    return basis_set

def Basis_Loop(new_mol, state, coords, alpha_MOs, beta_MOs, sets):
    for basis in sets[1:]:
        old_mol = deepcopy(new_mol)
        new_mol = Molecule(input, coords, basis)
        alpha_MOs = Basis_Fitting.Basis_Fit(old_mol, alpha_MOs, basis)
        beta_MOs = Basis_Fitting.Basis_Fit(old_mol, beta_MOs, basis)
        alpha_MOs, beta_MOs = hartree_fock.do(system,new_mol,state,alpha_MOs, beta_MOs)
    return alpha_MOs, beta_MOs

def Excite(matrix,occupancy, NElectrons):
# This function permutes the an array give an list describing the
# orbital occupancy
# Note this does not change its argument matrix
    new_matrix = deepcopy(matrix)
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

#----------------------------------------------------------------#
#                        THE MAIN PROGRAM                        #
#----------------------------------------------------------------#

system = System(input)
coords = input.Coords
alpha_MO_list = []
beta_MO_list = []
sets = map(remove_punctuation,input.BasisSets)

# do ground state calculation in the starting basis
molecule = Molecule(input, coords, sets[0])
base_alpha_MOs, base_beta_MOs = hartree_fock.do(system, molecule)
alpha_MO_list.append(base_alpha_MOs)
beta_MO_list.append(base_beta_MOs)

#Generate starting orbitals for each of the requested exicted states and do first calculation
for i, state in enumerate(molecule.States[1:], start = 1):
    alpha_MO_list.append(Excite(base_alpha_MOs, state.AlphaOccupancy, molecule.NAlphaElectrons))
    beta_MO_list.append(Excite(base_beta_MOs, state.BetaOccupancy, molecule.NBetaElectrons))
    alpha_MOs, beta_MOs = hartree_fock.do(system, molecule, alpha_MO_list, beta_MO_list, i)
    alpha_MO_list[-1] = alpha_MOs
    beta_MO_list[-1] = beta_MOs

# Possibly could also keep a list of the energies of each state and throw an
# error if they are not ordered at the end.
for basis_set in sets[1:]:
    # Iterate over list and preform basis fitting on each state
    for i in range(len(alpha_MO_list)):
        alpha_MO_list[i] = Basis_Fitting.Basis_Fit(molecule, alpha_MO_list[i], basis_set)
        beta_MO_list[i] = Basis_Fitting.Basis_Fit(molecule, beta_MO_list[i], basis_set)

    # Iterate over the list of states doing calculations while enforcing orthoginality
    molecule = Molecule(input, coords, basis_set)
    for i in range(len(alpha_MO_list)):
        alpha_MOs, beta_MOs = hartree_fock.do(system, molecule, alpha_MO_list, beta_MO_list, i)
        alpha_MO_list[i] = alpha_MOs
        beta_MO_list[i] = beta_MOs

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


def single_excitations(occupyed, unoccupyed):
    """Takes the strings of ground state occupyed and unoccupyed orbitals for a single
    spin and generates all possible single excitations between them"""
    excitations = []
    for i in range(-len(occupyed),0):
        for j in range(0,len(unoccupyed)):
            occupy = deepcopy(occupyed)
            unoccupy = deepcopy(unoccupyed)
            occupy[i] = 0
            unoccupy[j] = 1
            excitations.append(occupy+unoccupy)            
            
    return excitations


#----------------------------------------------------------------
# Set up data structures as python classes
# Convention: class variables are capitalized, instances not
#----------------------------------------------------------------

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
############ DIIS Settings #############
        try:
           self.UseMOM = input.UseMOM
        except:
            self.UseMOM = False 
        try:
            self.UseDIIS = input.UseDIIS 
        except:
            self.UseDIIS = True 
        try:
            self.DIIS_Size = input.DIIS_Size
        except:
            self.DIIS_Size = 15
        try:
            self.DIIS_Type = input.DIIS_Type 
        except:
            self.DIIS_Type = "C1"

############ Initial Guess Settings ##########
        try:
            #will need to add code to check if the data is avalible for SAD guess
            # for the givej molecule and basis
            self.SCFGuess = input.SCFGuess.lower() 
        except: 
            self.SCFGuess = 'core'

############ MOM Settings 
        try:
            self.UseMOM = input.UseMOM
        except:
            self.UseMOM = False 
        try:
            input.Excitations
            UseMOM = True
        except:
            UseMOM = False 
        try:
            # at this point the code just defaults to using the previous orbials as 
            # the reference at each iteration, will need to change the initialzation
            # to check for the 'fixed' keyword to used fixed reference orbitals
            self.MOM_Type = input.MOM_Type
        except: 
            self.MOM_Type = "mutable"
        self.out = Output.PrintSettings() 

class Molecule:
    def __init__(self,input,coords,basis_set):
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

#   !!!!!!!!!!! Need to make this block more general for unrestricted calculations !!!!!!!!!!!
       
        self.States = [ElectronicState(alpha_occupancy,beta_occupancy)]
        try:
            excitations = input.Excitations
            try:
                assert type(excitations) is list
                try:
                    for excitation in excitations:
                        assert type(excitation) is list 
                        occupyed = deepcopy(alpha_occupancy)
                        occupyed[n_alpha+excitation[0]] = 0
                        occupyed[n_alpha + excitation[1] - 1] = 1
                        self.States += [(ElectronicState(occupyed, beta_occupancy))]
                except AssertionError:
                    print 'Incorrect manual specification of single excitations'
                    print 'Require list of pairs of excitations, each pair is a list of 2 elements'
                    sys.exit() 
            except AssertionError:                         #If no list of excitations is provided
                if excitations == 'Single':
                    alpha_occupied = alpha_occupied
                elif excitations == 'Double':
                    # generate all strings corresponding to allowed double excitations
                    alpha_occupied = alpha_occupied
                else:
                    print 'Inappropriate value supplied for Excitations keyword'
                    sys.exit()
        except:
            pass

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
        basis_data = basis.get[basis_set]
        atom_basis_data = basis_data[self.Label]
        for function in atom_basis_data:
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

#----------------------------------------------------------------
#                       THE MAIN PROGRAM
#---------------------------------------------------------------

def remove_punctuation(basis_set):
    basis_set = basis_set.replace('*','s').replace('-','').replace('(','').replace(')','').replace(',','').upper()
    return basis_set



system = System(input)
coords = input.Coords
molecules = []
alpha_reference  = [[None]]     #this initiation avoids comparing an array to a single value latter in the code
beta_reference = [[None]]
sets = input.BasisSets
n_sets = len(sets)

#iterates over the list of sets
for i in xrange(n_sets):
    molecule = (Molecule(input,coords,remove_punctuation(sets[i])))

    for state in molecule.States:
        #alpha_MOs, beta_MOs = hartree_fock.do(system,molecule,state,alpha_reference, beta_reference)
        alpha_MOs, beta_MOs = cProfile.run('hartree_fock.do(system,molecule,state,alpha_reference, beta_reference)') # For profiling 
        system.DIIS = False 
        if i < n_sets - 1:       #does not preform the basis fitting on the final loop
            alpha_reference = Basis_Fitting.Basis_Fit(molecule, alpha_MOs,sets[i+1])
            beta_reference = Basis_Fitting.Basis_Fit(molecule, beta_MOs,sets[i+1])
        else:
            alpha_reference = alpha_MOs 
            beta_reference = beta_MOs
            
            



#for basis_set in input.BasisSets:    
#    molecule = (Molecule(input,coords,remove_punctuation(basis_set)))
#    for atom in molecule.Atoms:
#        for cgtf in atom.Basis:
#            print cgtf.Primitives
#        sys.exit()
#    MOs = hartree_fock.do(system,molecule,state,reference_orbitals)
#    reference_orbitals = Basis_Fitting.Basis_Fit(molecule )
    
# Check data structures are set up correctly
#print system.Method
#print molecule.Charge
#for atom in molecule.Atoms:
#    print atom.Label
#    for cgtf in atom.Basis:
#        print cgtf.AngularMomentum
#        for [exponent,coefficient] in cgtf.Primitives:
#            print exponent,coefficient

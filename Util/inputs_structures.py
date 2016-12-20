# System libraries
from __future__ import print_function
import sys
if sys.version_info.major is 2:
  import ConfigParser
else:
  import configparser as ConfigParser
import ast
import numpy
import math
import copy
# Custom-written data module
from Data import basis
import Data.constants as c
# Custom-written code modules
from Util import util
from Methods.hartree_fock import make_density_matrices

#=====================================================================#
#  MAIN ROUTINES - called from pychem.py and hartree_fock.py          #
#=====================================================================#
def process_input(section, parser):
    inputs = inputs_return_function(section, parser)
    settings = Settings(inputs)
    molecule = Molecule(inputs, settings)
    return molecule,settings

def inputs_return_function(section, parser):
    def inputs(var_name):
        var = parser.get(section,var_name)
        var = ast.literal_eval(var)
        if isinstance(var, str):
            var = var.upper()
        return var
    return inputs

#=====================================================================#
#  Global data structures, used absolutely everywhere                 #
#  Convention: class variables are capitalized, instances not         #
#              sub-routine names also not capitalized                 #
#  ------------------------------------------------------------------ #
#  Settings contains molecule-independent input data                  #
#  ------------------------------------------------------------------ #
#  Molecule contains molecule-dependent derived data                  #
#    -> each Molecule is made up of Atoms                             #
#       -> each Atom is described by a set of ContractedGaussians     #
#    -> each Molecule may exist in a range of ground/excited states   #
#  ------------------------------------------------------------------ #
#  AtomInMolecule is a molecule-like structure for an individual atom #
#  (as required for SAD guess)                                        #
#=====================================================================#

#=====================================================================#
#                             SETTINGS                                #
#=====================================================================#
class Settings:
    def __init__(self, inputs):
        #-------------------------------------------------------------#
        #               Universal settings subroutine                 #
        #-------------------------------------------------------------#
        self.set_universal(inputs)
        #-------------------------------------------------------------#
        #                  SCF settings subclasses                    #
        #-------------------------------------------------------------#
        self.SCF = Set_SCF(inputs,len(self.BasisSets),self.Method)
        self.DIIS = Set_DIIS(inputs,self.SCF.Reference)
        self.MOM = Set_MOM(inputs)
        self.NOCI = Set_NOCI(inputs)

    #=================================================================#
    def set_universal(self, inputs):
        #------------------------- Method ----------------------------#
        available_methods = ['HF','MP2']
        try:
           self.Method = inputs("Method")
           assert self.Method in available_methods
        except:
           print('Method either not specified or not recognised, defaulting to HF')
           print('Available methods:', available_methods)
           self.Method = 'HF'
        #------------------------ Job Type ---------------------------#
        available_jobtypes = ['ENERGY']
        try:
           self.JobType = inputs("Job_Type")
           assert self.JobType in available_jobtypes
        except:
#           print('Job type either not specified or not recognised, defaulting to single point energy')
#           print('Available job types:', available_jobtypes)
           self.JobType = 'ENERGY'
        #------------------------ Basis Set --------------------------#
        available_basis_sets = basis.get.keys()
        try:
           basis_sets = inputs("Basis_Sets")
           # If keyword found, check that input is in appropriate format
           try:
              assert type(basis_sets) is list
              assert type(basis_sets[0]) is str
              assert len(basis_sets) > 0
           except AssertionError:
              print('Error: basis sets must be specified as strings in a list, at least one must be given')
              sys.exit()
           basis_sets_clean = [util.remove_punctuation(basis_set) for basis_set in basis_sets]
           self.BasisSets = basis_sets_clean
           # Check that nominated basis sets are available
           for basis_set in self.BasisSets:
              try:
                 assert basis_set in available_basis_sets
              except AssertionError:
                 print('Error: basis set ', basis_set, ' not recognised')
                 sys.exit()
        except:
           print('Error: must include Basis_Sets keyword followed by at least one basis set in a list')
           sys.exit()
        #---------------------- Print Settings -----------------------#
        available_print_levels = ['MINIMAL','BASIC','VERBOSE','DEBUG', 'FINAL']
        available_custom_print_options = ['MOM','DIIS']
        try:
           self.PrintLevel = inputs("Print_Level")
           assert self.PrintLevel in available_print_levels
        except:
#           print('Printing options not supplied or recognised, defaulting to basic printing')
           self.PrintLevel = 'BASIC'
        try:
           self.CustomPrint = inputs("Custom_Print")
           assert self.CustomPrint in available_custom_print_options
        except:
#           print('Custom print options not supplied or recognised, defaulting to basic printing')
           pass
        try:
            self.PrintToTerminal = inputs("Print_To_Terminal")
            assert(isinstance(self.PrintToTerminal, bool))
        except:
            self.PrintToTerminal = False

    #=================================================================#
    #                 Set output file from pychem.py                  #
    def set_outfile(self, section_name):
        self.OutFileName = str(section_name) + '.out'
        self.OutFile = None
        self.SectionName = section_name
#=====================================================================#

#=====================================================================#
#          SETTINGS Subclasses - Set_SCF, Set_DIIS, Set_MOM           #
#=====================================================================#
class Set_SCF:
    def __init__(self, inputs, NBasisSets, Method):
        #------------------------- Reference -------------------------#
        available_references = ['RHF','UHF','CUHF']
        try:
           self.Reference = inputs("Reference")
        except:
           print('Reference not specified, defaulting to UHF')
           self.Reference = "UHF"
        try:
           assert self.Reference in available_references
        except AssertionError:
           print('Only the following references are available:', available_references)
           sys.exit()
        #------------------------- SCF Guess -------------------------#
        available_guesses = ['READ', 'CORE', 'SAD']
        try:
            self.Guess = inputs("SCF_Guess")
            assert(self.Guess in available_guesses)
        except:
            print("Could not identify SCF guess keyword, defaulting to core guess")
            self.Guess = "CORE"
        if self.Guess == "READ":
            try:
                self.MOReadName = inputs("MO_Read_Name")
                self.MOReadBasis = inputs("MO_Read_Basis")
            except:
                print("Must give appropriate details of file to read MOs from, using keywords MO_Read_Name and MO_Read_Basis")
                sys.exit()
        if self.Guess == "CORE":
            try:
                self.Guess_Mix = inputs("SCF_Guess_Mix")
            except ConfigParser.NoOptionError:
                self.Guess_Mix = 0
        #------------------------- SCF Cycles ------------------------#
        try:
            self.MaxIter = inputs("Max_SCF_Iterations")
        except:
            self.MaxIter = 25
        #------------------- Basis Fitting Switch --------------------#
        try:
           self.BasisFit = inputs("Basis_Fit")
        except:
           if NBasisSets == 1:
              self.BasisFit = False
           else:
              self.BasisFit = True
        try:
            assert type(self.BasisFit) is bool
        except AssertionError:
            print('Error: Basis_Fit must be set to True/False')
            sys.exit()
        #-------------- 2 Electron Integral Storage ------------------#
        available_handling_options = ['DIRECT','INCORE','ONDISK']
        try:
            self.Ints_Handling = inputs("2e_Ints_Handling")
            assert self.Ints_Handling in available_handling_options
        except:
            # Make INCORE storage default (fastest)
            self.Ints_Handling = 'INCORE'

#---------------------------------------------------------------------#
class Set_DIIS:
    def __init__(self,inputs,reference):
        try:
            self.Use = inputs("Use_DIIS")
        except:
            self.Use = True
        try:
            self.Size = inputs("DIIS_Size")
        except:
            self.Size = 15
        available_DIIS_types = ["C1","C2"]
        try:
            self.Type = inputs("DIIS_Type")
            if self.Type not in available_DIIS_types:
                Type = "C1"
        except:
            self.Type = "C1"
        try:
            self.Start = inputs("DIIS_Start")
        except:
            self.Start = 1
        try:
            self.MaxCondition = inputs("DIIS_Max_Condition")
        except:
            self.MaxCondition = c.DIIS_max_condition
        self.Threshold = 0.0

#---------------------------------------------------------------------#
class Set_MOM:
    def __init__(self,inputs):
        available_MOM_references = ["MUTABLE","FIXED"]
        try:
            self.Use = inputs("Use_MOM")
            assert type(self.Use) is bool
        except:
            try:
                inputs("Excitations")
                self.Use = True
            except:
                try:
                    inputs("Alpha_Excitations")
                    self.Use = True
                except:
                    self.Use = False
        try:
            self.Reference = inputs("MOM_Reference")
            assert self.Reference in available_MOM_references
        except:
            self.Reference = "FIXED"


#---------------------------------------------------------------------#
class Set_NOCI:
    def __init__(self, inputs):
        try:
            self.Use = inputs("Use_NOCI")
            assert type(self.Use) is bool
        except:
            self.Use = False

        try:
            self.print_level = inputs("NOCI_Print_Level")
        except:
            self.print_level = 1

#=====================================================================#

#=====================================================================#
#                             MOLECULE                                #
#=====================================================================#
class Molecule:
    def __init__(self, inputs, settings):
        #-------------------------------------------------------------#
        #  General molecule specifications - basis set independent    #
        #-------------------------------------------------------------#
        self.set_general(inputs)
        #-------------------------------------------------------------#
        #  Set up data structures and excitations for minimal basis   #
        #-------------------------------------------------------------#
        self.set_initial(inputs, settings)
        self.set_excitations(inputs)

    #=================================================================#
    def set_general(self, inputs):
        #----------------------- Coordinates -------------------------#
        if inputs != None:
           try:
              coords = inputs("Coords")
           except:
              print('Error: coordinates not found in input file')
              sys.exit()
           try:
              assert type(coords) is list
              assert type(coords[0]) is list
              assert len(coords[0]) == 5
           except AssertionError:
              print('Coordinates must be specified in list of lists format')
              print('Each atom entry is of form [atom_symbol,atom_nuc_charge,x,y,z]')
              sys.exit()
           self.NAtom = len(coords)
        #------------------------- Charge ----------------------------#
        if inputs != None:
           try:
              self.Charge = inputs("Charge")
           except:
              print('Error: must specify molecule charge using Charge =')
              sys.exit()
        #---------------------- Multiplicity -------------------------#
        if inputs != None:
           try:
              self.Multiplicity = inputs("Multiplicity")
           except:
              print('Error: must specify molecule multiplicity using Multiplicity =')
              sys.exit()
        #------------ Ground State Electronic Configuration -----------#
        self.NElectrons = 0
        self.NCoreOrbitals = 0
        for [label,Z,x,y,z] in coords:
            self.NElectrons += c.nElectrons[label.upper()]
            self.NCoreOrbitals += c.nCoreOrbitals[label.upper()]
        self.NElectrons = self.NElectrons - self.Charge
        try:
            self.NAlphaElectrons = int((self.NElectrons + (self.Multiplicity-1))/2)
            self.NBetaElectrons = int((self.NElectrons - (self.Multiplicity-1))/2)
        except:
            print('Error: charge and multiplicity inconsistent with specified molecule')
            sys.exit()

    #=================================================================#
    def set_initial(self, inputs, settings):
        #------------------------ Basis Set --------------------------#
        self.Basis = settings.BasisSets[0]
        #------------------------- Atoms List ------------------------#
        #       contains Atom -> ContractedGaussian subclasses        #
        #-------------------------------------------------------------#
        self.Atoms = []
        self.NOrbitals = 0
        index = 0
        coords = inputs("Coords")
        for row in coords:
            ### Add Atom to Molecule ###
            atom = Atom(index,row,self.Basis)
            self.Atoms.append(atom)
            self.NOrbitals += atom.NFunctions
            index += 1
        #---------- SCF structures - common to all states ------------#
        #          contains ShellPair -> Shell subclasses             #
        #-------------------------------------------------------------#
        Store2eInts = (settings.SCF.Ints_Handling == 'INCORE')
        self.initialize_intermediates(Store2eInts)

    def initialize_intermediates(self, Store2eInts):
        self.Core = numpy.zeros((self.NOrbitals,) * 2)
        self.Overlap = numpy.zeros((self.NOrbitals,) * 2)
        self.NuclearRepulsion = None
        self.X = []
        self.Xt = []
        self.S = []
        if Store2eInts:
            self.CoulombIntegrals = numpy.zeros((self.NOrbitals,) * 4)
            self.ExchangeIntegrals = numpy.zeros((self.NOrbitals,) * 4)
        ### Generate ShellPair data for Molecule ###
        self.make_shell_pairs()

    def make_shell_pairs(self):
        ia = -1
        ia_count = 0
        shell_pairs = []
        for atom_a in self.Atoms:
           for cgtf_a in atom_a.Basis:
              ia += 1
              ia_vec = [(ia_count + i) for i in range(0,cgtf_a.NAngMom)]   #vector contaning the indices each angular momentum function on the atom
              ia_count += cgtf_a.NAngMom                                   #total number of orbitals
              ib_count = 0
              ib = -1
              for atom_b in self.Atoms:
                 for cgtf_b in atom_b.Basis:
                    ib_vec = [(ib_count + i) for i in range(0,cgtf_b.NAngMom)]
                    ib_count += cgtf_b.NAngMom
                    shell_pair = ShellPair(atom_a.Coordinates,cgtf_a,ia,ia_vec,atom_b.Coordinates,cgtf_b,ib,ib_vec)   #forming all possible shell pairs
                    shell_pairs.append(shell_pair)
        self.ShellPairs = shell_pairs

    #=================================================================#
    def set_excitations(self, inputs):
        #------------- Excited state specifications ------------------#
        #      contains ElectronicState -> Matrices subclasses        #
        #-------------------------------------------------------------#
        available_excitation_types = ['SINGLE', 'DOUBLE', 'HOMO-LUMO', 'DOUBLE-PAIRED', 'CUSTOM-SINGLE','CUSTOM-PAIRED']
        try:
           excitation_type = inputs("Excitations")
        except ConfigParser.NoOptionError:
           try:
              alpha_excitations = inputs("Alpha_Excitations")
           except ConfigParser.NoOptionError:
              alpha_excitations = [[]]
           try:
              beta_excitations = inputs("Beta_Excitations")
           except ConfigParser.NoOptionError:
              beta_excitations = [[]]
           if (alpha_excitations == [[]]) and (beta_excitations == [[]]):
              excitation_type = None
           elif (alpha_excitations != [[]]) and (beta_excitations != [[]]):
              try:
                 assert len(alpha_excitations) == len(beta_excitations)
                 excitation_type = 'CUSTOM-PAIRED'
              except AssertionError:
                 print("""Alpha_Excitations and Beta_Excitations lists must be equal in length if both keywords present,
                          blank sub-lists can be used as placeholders""")
                 sys.exit()
           else:
              excitation_type = 'CUSTOM-SINGLE'

        # Check that excitations were given in the correct format
        try:
           assert excitation_type in available_excitation_types
        except:
           print("""Excitations can be specified by keyword as Excitations = 'Single, 'Double', 'Homo-Lumo' or 'Double-Paired'""")
           sys.exit()

        if (excitation_type is 'CUSTOM-SINGLE') or (excitation_type == 'CUSTOM-PAIRED'):
           try:
              assert type(alpha_excitations) is list
              for alpha_excitation in alpha_excitations:
                 assert type(alpha_excitation) is list
              assert type(beta_excitations) is list
              for beta_excitation in beta_excitations:
                 assert type(beta_excitation) is list
           except AssertionError:
              print("""Custom excitations specified as Alpha_Excitations or Beta_Excitations lists of [from,to] orbital pairs""")
              sys.exit()

        # Make excitation lists for corresponding keywords
        if (excitation_type == 'SINGLE') or (excitation_type == 'DOUBLE'):
           self.AlphaExcitations = util.single_excitations(self.NCoreOrbitals, self.NAlphaElectrons, self.NAlphaElectrons, self.NOrbitals)
           self.BetaExcitations = util.single_excitations(self.NCoreOrbitals, self.NBetaElectrons, self.NBetaElectrons, self.NOrbitals)
        if (excitation_type == 'DOUBLE-PAIRED'):
           self.AlphaExcitations = util.single_excitations(self.NCoreOrbitals, self.NBetaElectrons, self.NAlphaElectrons, self.NOrbitals)
        if (excitation_type == 'HOMO-LUMO'):
           self.AlphaExcitations = util.single_excitations(self.NAlphaElectrons-1, self.NAlphaElectrons, self.NAlphaElectrons, self.NAlphaElectrons+1)
        if (excitation_type == 'CUSTOM-SINGLE') or (excitation_type == 'CUSTOM-PAIRED'):
           self.AlphaExcitations = alpha_excitations
           self.BetaExcitations = beta_excitations
        self.ExcitationType = excitation_type

        #-------------------------------------------------------------#
        #   Generate excited states, each an ElectronicState object   #
        #-------------------------------------------------------------#
        self.generate_excited_states()
        self.NStates = len(self.States)

    def generate_excited_states(self):
        # Make occupancy lists for the ground state
        alpha_occupied = [1 for i in range(0,self.NAlphaElectrons)]
        beta_occupied = [1 for i in range(0,self.NBetaElectrons)]
        alpha_unoccupied = [0 for i in range(0,self.NOrbitals-self.NAlphaElectrons)]
        beta_unoccupied = [0 for i in range(0,self.NOrbitals-self.NBetaElectrons)]
        alpha_occupancy = alpha_occupied + alpha_unoccupied
        beta_occupancy = beta_occupied + beta_unoccupied
        # Instantiate ElectronicState for ground state
        self.States = [ElectronicState(alpha_occupancy,beta_occupancy,self.NOrbitals)]
        # Generate occupancy lists and ElectronicState instances for excited states
        if self.ExcitationType is not None:
           self.States = self.make_excitations()

    def make_excitations(self):
        alpha_ground = self.States[0].AlphaOccupancy
        beta_ground = self.States[0].BetaOccupancy

        # Common loop to make the excitations for all cases except single beta excitations
        for (i,alpha_excitation) in enumerate(self.AlphaExcitations):
            if alpha_excitation != []:
               alpha_occupied = self.do_excitation(alpha_ground, alpha_excitation)
            else:
               if self.ExcitationType == 'CUSTOM-PAIRED':
                  alpha_occupied = alpha_ground
                  beta_occupied = self.do_excitation(beta_ground, self.BetaExcitations[i])
               else:
                  break
            if (self.ExcitationType == 'SINGLE') or (self.ExcitationType == 'HOMO-LUMO'):
               beta_occupied = beta_ground
            if self.ExcitationType == 'DOUBLE-PAIRED':
               beta_occupied = alpha_occupied[:]
            if self.ExcitationType == 'DOUBLE':
               for beta_excitation in BetaExcitations:
                  beta_occupied = self.do_excitation(beta_ground, beta_excitation)
                  self.States += [(ElectronicState(alpha_occupied, beta_occupied, self.NOrbitals))]
            else:
               beta_occupied = beta_ground
               self.States += [(ElectronicState(alpha_occupied, beta_occupied, self.NOrbitals))]

        # Do single beta excitations separately
        if (self.ExcitationType == 'SINGLE') or (self.ExcitationType == 'CUSTOM-SINGLE'):
            alpha_occupied = alpha_ground
            if ((self.ExcitationType == 'SINGLE') and (self.Multiplicity == 1) and (self.AlphaExcitations != [[]])):
               pass
            else:
               for beta_excitation in self.BetaExcitations:
                  if beta_excitation != []:
                     beta_occupied = self.do_excitation(beta_ground, beta_excitation)
                     self.States += [(ElectronicState(alpha_occupied, beta_occupied, self.NOrbitals))]
                  else:
                     break

        return self.States

    def do_excitation(self, ground_occ, excitation):
        occupied = copy.deepcopy(ground_occ)
        if excitation != []:
            occupied[excitation[0]] = 0
            occupied[excitation[1]] = 1
        else:
            print("""Warning: shouldn't be calling do_excitation without a specified excitation pair""")
        return occupied

    #=================================================================#
    #     Helper function to return coordinates in useful formats     #
    def get_coords(self):
        full_coords = []
        labelled_coords = []
        coords = []
        for atom in self.Atoms:
            [x,y,z] = atom.Coordinates
            coords.append([x*c.toAng, y*c.toAng, z*c.toAng])
            labelled_coords.append([atom.Label,x*c.toAng, y*c.toAng, z*c.toAng])
            full_coords.append([atom.Label,atom.NuclearCharge,x*c.toAng, y*c.toAng, z*c.toAng])
        return coords,labelled_coords,full_coords

    #=================================================================#
    #                       Basis set update                          #
    def update_basis(self,basis_set,Store2eInts):
        # update variables then structures
        self.NOrbitals = 0
        self.Basis = basis_set
        for atom in self.Atoms:
            atom.update_atomic_basis(basis_set)
            self.NOrbitals += atom.NFunctions
        self.initialize_intermediates(Store2eInts)
        new_states = []
        for state in self.States:
            alpha_occupied = state.AlphaOccupancy + [0]*(self.NOrbitals-len(state.AlphaOccupancy))
            beta_occupied = state.BetaOccupancy + [0]*(self.NOrbitals-len(state.BetaOccupancy))
            new_state = ElectronicState(alpha_occupied, beta_occupied, self.NOrbitals)
            new_states.append(new_state)
        self.States = new_states

#=====================================================================#
#                      MOLECULE SUBCLASS - ATOM                       #
#=====================================================================#

class Atom:
    def __init__(self,index,row,basis_set):
        [label,Z,x,y,z] = row
        self.Index = index
        self.Label = label.upper()
        self.NuclearCharge = Z
        self.Coordinates = [x*c.toBohr,y*c.toBohr,z*c.toBohr]
        self.update_atomic_basis(basis_set)
    def update_atomic_basis(self,basis_set):
        self.Basis = []
        self.NFunctions = 0
        self.MaxAng = 0
        basis_data = basis.get[basis_set][self.Label]
        for function in basis_data:
            self.Basis.append(ContractedGaussian(function))
            self.NFunctions += int((function[0]+1)*(function[0]+2)/2)
            if function[0] > self.MaxAng:
                self.MaxAng = function[0]
    def update_coords(self,xyz):
        self.Coordinates = xyz

#---------------------------------------------------------------------#
#                 ATOM SUBCLASS - CONTRACTED GAUSSIAN                 #
#---------------------------------------------------------------------#

class ContractedGaussian:
    def __init__(self,function):
        self.AngularMomentum = function[0]
        self.NAngMom = c.nAngMomFunctions[self.AngularMomentum]
        self.Primitives = function[1:]
        self.NPrimitives = len(self.Primitives)
        self.NFunctions = self.AngularMomentum * self.NAngMom

#=====================================================================#
#               MOLECULE SUBCLASS - ELECTRONIC STATE                  #
#=====================================================================#

class ElectronicState:
    def __init__(self,alpha_occupancy,beta_occupancy,n_orbitals):
        self.AlphaOccupancy = alpha_occupancy
        self.BetaOccupancy = beta_occupancy
        self.S2 = None
        self.Energy = None
        self.TotalEnergy = None
        total_occupancy = [alpha_occupancy[i] + beta_occupancy[i] for i in range(len(alpha_occupancy))]
        self.Total = Matrices(n_orbitals, total_occupancy, total=True)
        self.Alpha = Matrices(n_orbitals, alpha_occupancy)
        self.Beta = Matrices(n_orbitals, beta_occupancy)
#        self.Gradient = gradient
#        self.Hessian = hessian
        self.AlphaDIIS = StoreDIIS()
        self.BetaDIIS = StoreDIIS()

#---------------------------------------------------------------------#
#                ELECTRONIC STATE SUBCLASS - MATRICES                 #
#---------------------------------------------------------------------#

class Matrices:
    def __init__(self,n_orbitals,occupancy,total=False):
        self.Density = numpy.zeros((n_orbitals,) * 2)
        self.Fock = numpy.zeros((n_orbitals,) * 2)
        self.Occupancy = occupancy
        if not total:
            self.Exchange = numpy.zeros((n_orbitals,) * 2)
            self.MOs = []
            self.Energies = []
        else:
            self.Coulomb = numpy.zeros((n_orbitals,) * 2)

    def damp(self, coeff):
        try:
            self.old_Fock
        except AttributeError:
            self.old_Fock = copy.deepcopy(self.Fock)
        damped_Fock = self.old_Fock * coeff + self.Fock * (1 - coeff)
        self.old_Fock = copy.deepcopy(self.Fock)
        self.Fock = damped_Fock

#---------------------------------------------------------------------#
#                  ELECTRONIC STATE SUBCLASS - DIIS                   #
#---------------------------------------------------------------------#

class StoreDIIS:
    def __init__(self):
        self.OldFocks = []
        self.OldDensities = []
        self.Residuals = []
        self.Matrix = [[None]]
        self.Error = 1
        self.pre_DIIS_fock = [[None]]

#=====================================================================#
#                  MOLECULE SUBCLASS - SHELLPAIR                      #
#=====================================================================#

class ShellPair:
    def __init__(self,coords_a,cgtf_a,ia,ia_vec,coords_b,cgtf_b,ib,ib_vec):
       # precompute all ShellPair data
       # note that ia & ib are vectors of Fock matrix indices for each cgtf of length NAngMom
       self.Centre1 = Shell(coords_a,cgtf_a,ia,ia_vec)
       self.Centre2 = Shell(coords_b,cgtf_b,ib,ib_vec)

#---------------------------------------------------------------------#
#                    SHELLPAIR SUBCLASS - SHELL                       #
#---------------------------------------------------------------------#

class Shell:
    def __init__(self,coords,cgtf,index,index_vec):
       self.Coords = coords
       self.Cgtf = cgtf
       self.Index = index     # Index for the CGTO in atom.Basis
       self.Ivec = index_vec  # Indexs of the angular momentum functions in a list of all angular momentum functions on the atom

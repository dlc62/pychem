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
# Custom-written data modules
import basis
import constants as c
# Custom-written code modules
import util

#=====================================================================#
#  MAIN ROUTINES - called from pychem.py and hartree_fock.py          #
#=====================================================================#
def process_input(section, parser):
    inputs = inputs_return_function(section, parser)
    settings = Settings(inputs)
    molecule = Molecule(inputs, settings, basis=None)
    return molecule,settings

def inputs_return_function(section, parser):
    def inputs(var_name):
        var = parser.get(section,var_name)
        var = ast.literal_eval(var)
        if isinstance(var, str):
            var = var.upper()
        return var
    return inputs

def new_molecule(basis):
    molecule = Molecule(inputs=None, settings=None, basis)
    return molecule

#=====================================================================#
#  Global data structures, used absolutely everywhere                 #
#  Convention: class variables are capitalized, instances not         #
#  ------------------------------------------------------------------ #
#  Settings contains system-independent input data                    #
#  ------------------------------------------------------------------ #
#  Molecule contains system-dependent derived data                    #
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
        #                     Universal settings                      #
        #-------------------------------------------------------------#
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
           print('Job type either not specified or not recognised, defaulting to single point energy')
           print('Available job types:', available_jobtypes)
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
        available_print_levels = ['MINIMAL','BASIC','VERBOSE','DEBUG']
        available_custom_print_options = ['MOM','DIIS'] 
        try:
           self.PrintLevel = inputs("Print_Level")
           assert self.PrintLevel in available_print_levels
        except:
           print('Printing options not supplied or recognised, defaulting to basic printing') 
           self.PrintLevel = 'BASIC'
        try:
           self.CustomPrint = inputs("Custom_Print")
           assert self.CustomPrint in available_custom_print_options
        except:
           print('Custom print options not supplied or recognised, defaulting to basic printing') 
        #-------------------------------------------------------------#
        #                         SCF Settings                        #
        #-------------------------------------------------------------#
        self.SCF = Set_SCF(inputs,len(self.BasisSets),self.Method)
        self.DIIS = Set_DIIS(inputs,self.SCF.Reference)
        self.MOM = Set_MOM(inputs)
        
    def set_outfile(self, section_name):
        self.OutFileName = str(section_name) + '.out'
        self.OutFile = None
        self.SectionName = section_name

#=====================================================================#
#                      SETTINGS Subclass - Set_SCF                    #
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
            if Method == 'MP2':
                # Make INCORE storage default for MP2
                self.Ints_Handling = 'INCORE'
            else:
                # Choose the lowest memory option by default
                self.Ints_Handling = 'DIRECT'

#=====================================================================#
#                     SETTINGS Subclass - Set_DIIS                    #
#=====================================================================#

class Set_DIIS:
    def __init__(self,inputs,reference):
        if reference == "CUHF":
            self.Use = False
        else:
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

#=====================================================================#
#                     SETTINGS Subclass - Set_MOM                     #
#=====================================================================#

class Set_MOM:
    def __init__(self,inputs):
        #---------------------------- MOM ----------------------------#
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

#######################################################################

#=====================================================================#
#                             MOLECULE                                #
#=====================================================================#

class Molecule:
    def __init__(self, inputs, settings, basis = None):
        #--------------------------------------------------------------#
        #                    Molecular Specifications                  #
        #--------------------------------------------------------------#

        #------------------------- Basis Set --------------------------#
        if basis == None:
           self.Basis = settings.BasisSets[0]
        else: 
           self.Basis = basis

        #------------------------ Coordinates -------------------------#
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

        #-------------------------- Charge ----------------------------#
        if inputs != None:
           try:
              self.Charge = inputs("Charge")
           except:
              print('Error: must specify molecule charge using Charge =')
              sys.exit()

        #----------------------- Multiplicity -------------------------#
        if inputs != None:
           try:
              self.Multiplicity = inputs("Multiplicity")
           except:
              print('Error: must specify molecule multiplicity using Multiplicity =')
              sys.exit()

        #-------------------------- Atoms List ------------------------#
        #        contains Atom -> ContractedGaussian subclasses        #
        #--------------------------------------------------------------#
        self.Atoms = []
        n_electrons = 0
        n_orbitals = 0
        n_core_orbitals = 0
        index = 0
        for row in coords:
            ### Add Atom to Molecule ###
            atom = Atom(index,row,self.Basis)
            self.Atoms.append(atom)
            n_electrons += c.nElectrons[atom.Label]
            n_core_orbitals += c.nCoreOrbitals[atom.Label]
            n_orbitals += atom.NFunctions
            index += 1

        #------------ Ground State Electronic Configuration -----------#
        n_electrons = n_electrons - self.Charge
        try:
            n_alpha = (n_electrons + (self.Multiplicity-1))/2
            n_beta  = (n_electrons - (self.Multiplicity-1))/2
        except:
            print('Error: charge and multiplicity inconsistent with specified molecule')
            sys.exit()
        self.NElectrons = int(n_electrons)
        self.NAlphaElectrons = int(n_alpha)
        self.NBetaElectrons = int(n_beta)
        self.NOrbitals = int(n_orbitals)
        self.NAlphaOrbitals = int(math.ceil(n_alpha/2.0))
        self.NBetaOrbitals = int(math.ceil(n_alpha/2.0))

        #----------- SCF structures - common to all states ------------#
        #           contains ShellPair -> Shell subclasses             #    
        #--------------------------------------------------------------#
        if settings != None:
            self.Store2eInts = (settings.SCF.Ints_Handling == 'INCORE') 
            self.Recalc2eInts = (settings.SCF.Ints_Handling == 'DIRECT')
            self.Dump2eInts = (settings.SCF.Ints_Handling == 'ONDISK')
        self.Core = numpy.zeros((self.NOrbitals,) * 2)
        self.Overlap = numpy.zeros((self.NOrbitals,) * 2)
        self.NuclearRepulsion = None
        self.X = []
        self.Xt = []
        self.S = []
        if self.Store2eInts:
            self.CoulombIntegrals = numpy.zeros((self.NOrbitals,) * 4) 
            self.ExchangeIntegrals = numpy.zeros((self.NOrbitals,) * 4) 
        ### Generate ShellPair data for Molecule ###
        self.make_shell_pairs()

        #-------------- Excited state specifications ------------------#
        #       contains ElectronicState -> Matrices subclasses        #
        #--------------------------------------------------------------#
        if inputs != None:
           alpha_excitations = [[]]
           beta_excitations = [[]]
           try:
              alpha_excitations = inputs("Alpha_Excitations")
              beta_excitations = inputs("Beta_Excitations")
           except ConfigParser.NoOptionError:
              try:
                 alpha_excitations = inputs("Excitations")
              except ConfigParser.NoOptionError:
                 pass
           # Check that excitations were given in the correct format
           try:
              alpha_is_list = type(alpha_excitations) is list
              alpha_is_keyword = alpha_excitations in ['Single', 'Double']
              assert alpha_is_list or alpha_is_keyword
              if alpha_is_list:
                 assert type(beta_excitations) is list
           except AssertionError:
              print("""Excitations must be specified as a list of excitations each containing two elements
                       or using the keywords 'Single' or 'Double'""")
              sys.exit()
           self.AlphaExcitations = alpha_excitations
           self.BetaExcitations = beta_excitations
        ### Generate excited states, each an ElectronicState ###
        self.generate_excited_states()
        self.NStates = len(self.States)

    #================== MOLECULE CLASS SUBROUTINES ===================# 

    #------------------------------------------------------------------#
    #       Shell pairs required for all states including ground       #
    #------------------------------------------------------------------#

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

    #-------------------------------------------------------------------------#
    #       Generate orbital occupancy lists that define excited states       #
    #-------------------------------------------------------------------------#

    def generate_excited_states(self):
        alpha_occupied = [1 for i in range(0,self.NAlphaElectrons)]
        beta_occupied = [1 for i in range(0,self.NBetaElectrons)]
        alpha_unoccupied = [0 for i in range(0,self.NOrbitals-self.NAlphaElectrons)]
        beta_unoccupied = [0 for i in range(0,self.NOrbitals-self.NBetaElectrons)]
        # Combine the occupied and unoccupied lists to make two (alpha and beta) total occupancy lists
        alpha_occupancy = alpha_occupied + alpha_unoccupied  
        beta_occupancy = beta_occupied + beta_unoccupied
        ground = ElectronicState(alpha_occupancy,beta_occupancy,self.NOrbitals)
        self.States = self.make_excitations(ground)

    def make_excitations(self, ground):
        self.States = [ground]
        alpha_ground = ground.AlphaOccupancy
        beta_ground = ground.BetaOccupancy

        if self.AlphaExcitations == self.BetaExcitations == [[]]:
            # immediately returns if no excitations were specified
            return self.States
        elif alpha_excitations == 'Single':
            alpha_excitations = util.single_excitations(self.NAlphaElectrons, self.NOrbitals)
        elif alpha_excitations == 'Double':
            alpha_singles = util.single_excitations(self.NAlphaElectrons, self.NOrbitals)
            beta_singles = util.single_excitations(self.NBetaElectrons, self.NOrbitals)
            for excite1 in alpha_singles:
                for excite2 in beta_singles:
                    pass
        # else manual specification of orbitals

        util.make_length_equal(alpha_excitations, beta_excitations)
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
            self.States += [(ElectronicState(alpha_occupied, beta_occupied, self.NOrbitals))]
        return self.States

    def do_excitation(self, n_electrons, ground_occ, excitation):
        occupied = deepcopy(ground_occ)
        if excitation != []:
            occupied[n_electrons + excitation[0]] = 0
            occupied[n_electrons + excitation[1] - 1] = 1
        return occupied

    #-----------------------------------------------------------------#
    #     Helper function to return coordinates in useful formats     #
    #-----------------------------------------------------------------#

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

#    #-------------------------------------------------------------------------#
#    #                           Basis set update                              #
#    #-------------------------------------------------------------------------#
#    def update_basis(self,basis_set):
#        # update variables then structures
#        n_orbitals = 0
#        for atom in self.Atoms:
#            atom.update_atomic_basis(basis_set)
#            n_orbitals += atom.NFunctions
#        self.NOrbitals = n_orbitals
#        self.initialize_intermediates()
#        self.generate_excited_states()
        

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
            self.NFunctions += (function[0]+1)*(function[0]+2)/2
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
        self.Total = Matrices(n_orbitals,total=True)
        self.Alpha = Matrices(n_orbitals) 
        self.Beta = Matrices(n_orbitals) 
#        self.Gradient = gradient
#        self.Hessian = hessian
        self.AlphaDIIS = StoreDIIS()
        self.BetaDIIS = StoreDIIS()

#---------------------------------------------------------------------#
#                ELECTRONIC STATE SUBCLASS - MATRICES                 #
#---------------------------------------------------------------------#

class Matrices:
    def __init__(self,n_orbitals,total=False):
        self.Density = numpy.zeros((n_orbitals,) * 2)
        self.Fock = numpy.zeros((n_orbitals,) * 2)
        if not total:
            self.Exchange = numpy.zeros((n_orbitals,) * 2)
            self.MOs = []
            self.Energies = [] 
        else:
            self.Coulomb = numpy.zeros((n_orbitals,) * 2)

#---------------------------------------------------------------------#
#                  ELECTRONIC STATE SUBCLASS - DIIS                   #
#---------------------------------------------------------------------#

class StoreDIIS:
    def __init__(self):
        self.OldFocks = []
        self.Residuals = []
        self.Matrix = None
        self.Error = 1

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


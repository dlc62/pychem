# Import system libraries
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
# Import custom-written data modules
from Data import basis
from Data import constants as c
from Data import transform_basis
# Import custom-written code modules
from Methods import _c_ints

#=====================================================================#
#  MAIN ROUTINES - called from pychem.py and hartree_fock.py          #
#=====================================================================#
def process_input(section, parser):
    inputs = inputs_return_function(section, parser)
    settings = Settings(inputs, section)
    molecule = Molecule(inputs=inputs, settings=settings)
    if settings.SCF.BasisFit and (molecule.CartesianL != []):
       print('Error: basis fitting currently not available for Cartesian basis functions')
       sys.exit()
    return molecule,settings

def inputs_return_function(section, parser):
    def inputs(var_name):
        var = parser.get(section,var_name)
        var = ast.literal_eval(var)
        if isinstance(var, str):
            var = var.upper()
        return var
    return inputs

def update_basis(self,basis_set):
    self.Basis = basis_set
    self.set_structures()
    self.set_scf()
    self.ExcitationType = 'COPY'
    self.set_excitations()
    return self

#=====================================================================#
#  Global data structures, used absolutely everywhere                 #
#  Convention: class variables are capitalized, instances not         #
#  ------------------------------------------------------------------ #
#  Settings contains system-independent input data                    #
#  ------------------------------------------------------------------ #
#  Molecule contains system-dependent derived data                    #
#    -> each Molecule is made up of Atoms                             # 
#       -> each Atom is described by a set of ContractedGaussians     #
#         => pairs of these CGTFs form shell-pairs                    #
#           => shell-pairs used in calculating integrals              #
#              stored in integral arrays for each molecule            #
#    -> each Molecule may exist in a range of ground/excited states   #
#  ------------------------------------------------------------------ #
#  AtomInMolecule is a molecule-like structure for an individual atom #
#  (as required for SAD guess)                                        #
#=====================================================================#

#=====================================================================#
#                             SETTINGS                                #
#=====================================================================#
class Settings:
    def __init__(self, inputs, section_name):
        #-------------------------------------------------------------#
        #                     Universal settings                      #
        #-------------------------------------------------------------#
        #------------------------- Method ----------------------------#
        available_methods = [None,'HF','MP2','NOCI']
        try:
           self.Method = inputs("Method").upper()
           assert self.Method in available_methods
        except:
           print('Method either not specified or not recognised, defaulting to HF')
           print('Available methods:', available_methods)
           self.Method = 'HF'
        if self.Method == 'NOCI':
           try:
              inputs("Excitations")
           except:
              try:
                 inputs("Alpha_Excitations")
              except:
                 print('Error: must specify excited states for NOCI')
                 sys.exit()
        #------------------------ Job Type ---------------------------#
        available_jobtypes = ['ENERGY','PROPERTY']
        try:
           self.JobType = inputs("Job_Type").upper()
           assert self.JobType in available_jobtypes
        except:
           print('Job type either not specified or not recognised, defaulting to single point energy')
           print('Available job types:', available_jobtypes)
           self.JobType = 'ENERGY'
        #--------------------- Property Type -------------------------#
        available_properties = ['SCATTERING']
        if self.JobType == 'PROPERTY':
           try:
              self.PropertyType = inputs("Property_Type").upper()
              assert self.PropertyType in available_properties
              try: 
                 self.PropertyGrid = inputs("Property_Grid") 
                 assert type(self.PropertyGrid) is list
              except:
                 print('Error: Must give grid values at which to evaluate properties')
                 sys.exit()
           except:
              print('Requested property type not recognised/implemented')
              print('Available property types:', available_properties)
              self.PropertyType = None
        #----------------------- Basis Set ---------------------------#
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
           basis_sets_clean = [remove_punctuation(basis_set) for basis_set in basis_sets]
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
           self.PrintLevel = inputs("Print_Level").upper()
           assert self.PrintLevel in available_print_levels
        except:
        #   print('Printing options not supplied or recognised, defaulting to basic printing') 
           self.PrintLevel = 'BASIC'
        try:
           self.CustomPrint = inputs("Custom_Print").upper()
           assert self.CustomPrint in available_custom_print_options
        except:
        #   print('Custom print options not supplied or recognised, defaulting to basic printing') 
           pass
        try:
            self.DumpMOs = inputs("Dump_MOs")
            assert type(self.DumpMOs) is bool
        except:
            self.DumpMOs = False
        self.OutFileName = str(section_name) + '.out'
        self.OutFile = None
        self.SectionName = section_name
        #-------------------------------------------------------------#
        #                         SCF Settings                        #
        #-------------------------------------------------------------#
        self.SCF = Set_SCF(inputs,len(self.BasisSets),self.Method)
        self.DIIS = Set_DIIS(inputs,self.SCF.Reference)
        self.MOM = Set_MOM(inputs)
        
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
           print('Error: Only the following references are available:', available_references)
           sys.exit()
        try:
           self.ConstrainExcited = inputs("Constrain_Excited")
        except:
           if self.Reference == "CUHF":
               self.ConstrainExcited = True
           else:
               self.ConstrainExcited = False
        #------------------------- SCF Guess -------------------------#
        available_guesses = ['READ', 'CORE', 'SAD']
        #------------------------- SCF Guess -------------------------#
        available_guesses = ['READ', 'CORE', 'SAD']
        try:
            self.Guess = inputs("SCF_Guess")
            assert(self.Guess in available_guesses)
        except:
        #    print("Could not identify SCF guess keyword, defaulting to core guess")
            self.Guess = "CORE"
        if self.Guess == "READ" or Method == None:
            try:
                mo_read_state = inputs("MO_Read_State")
                if type(mo_read_state) is list:
                    mo_read_state = [str(i) for i in mo_read_state]
                mo_read_basis = remove_punctuation(inputs("MO_Read_Basis"))
                self.AlphaMOFile = [mo_read_basis + '_' + state + '.alpha_MOs' for state in mo_read_state]
                self.BetaMOFile = [mo_read_basis + '_' + state + '.beta_MOs' for state in mo_read_state]
            except:
                print("Error: Must give details of files to read alpha and beta MOs from, using keywords MO_Read_Basis and MO_Read_State")
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
 #       #-------------- 2 Electron Integral Storage ------------------#
 #       available_handling_options = ['DIRECT','INCORE','ONDISK']
 #       try:
 #           self.Ints_Handling = inputs("2e_Ints_Handling")
 #           assert self.Ints_Handling in available_handling_options
 #       except:
 #           # Choose the fastest option by default (only one actually supported at the moment)
 #           self.Ints_Handling = 'INCORE'

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

    def __init__(self, inputs = None, settings = None, basis = None):
        #--------------------------------------------------------------#
        #           Parse inputs and set up data structures            # 
        #          for ground and excited state calculations           #
        #--------------------------------------------------------------#
        if basis == None:
           self.parse_inputs(inputs)
           basis = settings.BasisSets[0]
        self.Basis = basis
        self.set_structures() 
        self.set_scf()
        self.set_excitations()

    #==================================================================#
    def parse_inputs(self,inputs):
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
              print('Error: Coordinates must be specified in list of lists format')
              print('Each atom entry is of form [atom_symbol,atom_nuc_charge,x,y,z]')
              sys.exit()
           self.Coords = coords
           self.NAtom = len(coords)
           available_units = ['ANGSTROM','BOHR','ATOMIC']
           coords_scale_factor = 1.0
           try:
              coords_units = inputs("Coords_Units").upper()
              assert coords_units in available_units
              if coords_units == 'ANGSTROM': coords_scale_factor = c.toBohr
           except:
              coords_scale_factor = c.toBohr
           self.CoordsScaleFactor = coords_scale_factor

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

        #-------- Special flags for basis set modifications -----------#
        try:
           cartesian_l = inputs("Cartesian_L")
           self.CartesianL = cartesian_l
        except:
           self.CartesianL = []

        try:
           max_l = inputs("Max_L")
           self.MaxL = max_l
        except:
           self.MaxL = 100   # Ridiculously large dummy value

        #----------------------- Excitations --------------------------#
        available_excitation_types = [None, 'HOMO-LUMO', 'CUSTOM-SINGLE','CUSTOM-SINGLES', 'CUSTOM-PAIRED',
                                      'SINGLE', 'DOUBLE', 'SINGLE-DOUBLE', 'DOUBLE-PAIRED', 
                                      'SINGLES', 'DOUBLES', 'SINGLES-DOUBLES', 'DOUBLES-PAIRED', 
                                      'SPIN-FLIP-SINGLE','SPIN-FLIP-DOUBLE', 'SPIN-FLIP-SINGLE-DOUBLE',
                                      'SPIN-FLIP-SINGLES','SPIN-FLIP-DOUBLES', 'SPIN-FLIP-SINGLES-DOUBLES',
                                      'S','D','SD','SF-S','SF-D','SF-SD','SFS','SFD','SFSD',
                                      'SFS-CISD','SFD-CISD','SFD-CISDT','SFD-CISDTQ',
                                      'SFSD-CISD','SFSD-CISDT','SFSD-CISDTQ']

        try:
           excitation_type = inputs("Excitations").upper()
           if (excitation_type == 'CUSTOM-SINGLES'): excitation_type = 'CUSTOM-SINGLE'
           if (excitation_type == 'DOUBLES-PAIRED'): excitation_type = 'DOUBLE-PAIRED'
           if (excitation_type == 'SINGLE') or (excitation_type == 'SINGLES'): excitation_type = 'S'
           if (excitation_type == 'DOUBLE') or (excitation_type == 'DOUBLES'): excitation_type = 'D'
           if (excitation_type == 'SINGLE-DOUBLE') or (excitation_type == 'SINGLES-DOUBLES'): excitation_type = 'SD'
           if (excitation_type == 'SPIN-FLIP-SINGLE') or (excitation_type == 'SPIN-FLIP-SINGLES') or (excitation_type == 'SF-S'): excitation_type = 'SFS'
           if (excitation_type == 'SPIN-FLIP-DOUBLE') or (excitation_type == 'SPIN-FLIP-DOUBLES') or (excitation_type == 'SF-D'): excitation_type = 'SFD'
           if (excitation_type == 'SPIN-FLIP-SINGLE-DOUBLE') or (excitation_type == 'SPIN-FLIP-SINGLES-DOUBLES') or (excitation_type == 'SF-SD'): excitation_type = 'SFSD'
        except ConfigParser.NoOptionError:
           # Look for custom list of alpha excitations
           try:
              alpha_excitations = inputs("Alpha_Excitations")
           except ConfigParser.NoOptionError:
              alpha_excitations = [[]]
           # Look for custom list of beta excitations
           try:
              beta_excitations = inputs("Beta_Excitations")
           except ConfigParser.NoOptionError:
              beta_excitations = [[]]
           # If neither are specified, then no excitations are possible
           if (alpha_excitations == [[]]) and (beta_excitations == [[]]):
              excitation_type = None
           # If both are specified, they must be of equal length
           elif (alpha_excitations != [[]]) and (beta_excitations != [[]]):
              try:
                 assert len(alpha_excitations) == len(beta_excitations)
                 excitation_type = 'CUSTOM-PAIRED'
              except AssertionError:
                 print("""Error: Alpha_Excitations and Beta_Excitations lists must be equal in length if both keywords present,
                          blank sub-lists can be used as placeholders to specify unpaired excitations""")
                 sys.exit()
           # Otherwise only one or the other is specified
           else:
              excitation_type = 'CUSTOM-SINGLE'

        ### Check that excitations were given in the correct format ###
        try:
           assert excitation_type in available_excitation_types
        except:
           print("Error: Excitations can be specified by keyword as Excitations = specifier")
           print("Available specifiers: ", available_excitation_types)
           sys.exit()

        if (excitation_type is 'CUSTOM-SINGLE') or (excitation_type == 'CUSTOM-PAIRED'):
           try:
              assert type(alpha_excitations) is list
              for alpha_excitation in alpha_excitations:
                 assert type(alpha_excitation) is list
              assert type(beta_excitations) is list
              for beta_excitation in beta_excitations:
                 assert type(beta_excitation) is list
              self.AlphaExcitations = alpha_excitations
              self.BetaExcitations = beta_excitations
           except AssertionError:
              print("""Error: Custom excitations specified as Alpha_Excitations or Beta_Excitations lists of [from,to] orbital pairs""")
              sys.exit()

        self.ExcitationType = excitation_type

    #==================================================================#
    def set_structures(self):
        #-------------------------- Atoms List ------------------------#
        #        contains Atom -> ContractedGaussian subclasses        #
        #--------------------------------------------------------------#
        self.Atoms = []
        self.NElectrons = 0
        self.NOrbitals = 0
        self.NCoreOrbitals = 0
        self.NCgtf = 0
        for index,row in enumerate(self.Coords):
            ### Add Atom to Molecule ###
            atom = Atom(index,row,self.Basis,self.CartesianL,self.MaxL,self.CoordsScaleFactor)
            self.Atoms.append(atom)
            self.NElectrons += c.nElectrons[atom.Label]
            self.NCoreOrbitals += c.nCoreOrbitals[atom.Label]
            self.NOrbitals += atom.NFunctions
            self.NCgtf += len(atom.Basis) 

        #------------ Ground State Electronic Configuration -----------#
        self.NElectrons -= self.Charge
        try:
            self.NAlphaElectrons = (self.NElectrons + (self.Multiplicity-1))/2
            self.NBetaElectrons  = (self.NElectrons - (self.Multiplicity-1))/2
        except:
            print('Error: charge and multiplicity inconsistent with specified molecule')
            sys.exit()
        self.NAlphaOrbitals = int(math.ceil(self.NAlphaElectrons/2.0))
        self.NBetaOrbitals = int(math.ceil(self.NBetaElectrons/2.0))

    def set_scf(self):
        #----------- SCF structures - common to all states ------------#
        #           contains ShellPair -> Shell subclasses             #    
        #--------------------------------------------------------------#
        self.Core = numpy.zeros((self.NOrbitals,) * 2)
        self.Overlap = numpy.zeros((self.NOrbitals,) * 2)
        self.NuclearRepulsion = None
        self.X = []
        self.Xt = []
        self.S = []
        self.Si = []
        self.Bounds = numpy.ndarray.tolist(numpy.zeros((self.NCgtf,) * 2)) 
        self.CoulombIntegrals = numpy.zeros((self.NOrbitals,) * 4) 
        ### Generate and store ShellPair data for all unique pairs of CGTFs ###
        ### Include data required to convert basis functions and integrals  ###
        ### from Cartesian to spherical polar coordinates before storing    ### 
        shell_pairs = {}
        ia = -1; ia_count = 0
        for atom_a in self.Atoms:
           for cgtf_a in atom_a.Basis:
              ia_vec = [(ia_count + i) for i in range(0,cgtf_a.NAngMom)]
              ia += 1; ia_count += cgtf_a.NAngMom
              ib = -1; ib_count = 0
              for atom_b in self.Atoms:
                 for cgtf_b in atom_b.Basis:
                    ib_vec = [(ib_count + i) for i in range(0,cgtf_b.NAngMom)]
                    ib += 1; ib_count += cgtf_b.NAngMom
                    if ib >= ia:
                       shell_pairs[(ia,ib)] = ShellPair(atom_a.Coordinates,cgtf_a,ia,ia_vec,atom_b.Coordinates,cgtf_b,ib,ib_vec)
        self.ShellPairs = shell_pairs

    #==================================================================#
    def set_excitations(self):
        #-------------- Excited state specifications ------------------#
        #       contains ElectronicState -> Matrices subclasses        #
        #--------------------------------------------------------------#

        # Copy states generated in smaller (e.g. minimal) basis
        # Note that basis fitting/bootstrapping is NOT compatible with spin-flipping
        # which should be done straight-up in desired basis set

        if self.ExcitationType == 'COPY':

            new_states = []
            
            for state in self.States:
               new_alpha = state.Alpha.Occupancy + [0 for i in range(0,self.NOrbitals-len(state.Alpha.Occupancy))] 
               new_beta  = state.Beta.Occupancy +  [0 for i in range(0,self.NOrbitals-len(state.Beta.Occupancy))] 
               new_states.append(ElectronicState(new_alpha,new_beta,self.NOrbitals))
            
            self.States = new_states
            self.NStates = len(new_states)
            self.SpinFlipStates = []
             
            return

        ### Make occupancy lists for the ground state ###
        alpha_ground = [1 for i in range(0,self.NAlphaElectrons)] + [0 for i in range(0,self.NOrbitals-self.NAlphaElectrons)]
        beta_ground = [1 for i in range(0,self.NBetaElectrons)] + [0 for i in range(0,self.NOrbitals-self.NBetaElectrons)]

        ### Instantiate ElectronicState for ground state ###
        self.States = [ElectronicState(alpha_ground,beta_ground,self.NOrbitals)]
        
        if self.ExcitationType is None:
            self.ExcitationType = "None"
            self.NStates = 1
            self.SpinFlipStates = []
            return

        ### Generate general excitations lists or spin-flip excitation lists and electronic states (interdependent) ###
        spin_flip_states = []
        if (self.ExcitationType == 'DOUBLE-PAIRED'):
           self.AlphaExcitations = self.single_excitations(self.NCoreOrbitals, self.NBetaElectrons, self.NAlphaElectrons, self.NOrbitals)
        elif (self.ExcitationType == 'HOMO-LUMO'):
           self.AlphaExcitations = self.single_excitations(self.NAlphaElectrons-1, self.NAlphaElectrons, self.NAlphaElectrons, self.NAlphaElectrons+1)
        elif (self.ExcitationType == 'S') or (self.ExcitationType == 'D') or (self.ExcitationType == 'SD'):
           self.AlphaExcitations = self.single_excitations(self.NCoreOrbitals, self.NAlphaElectrons, self.NAlphaElectrons, self.NOrbitals)
           self.BetaExcitations = self.single_excitations(self.NCoreOrbitals, self.NBetaElectrons, self.NBetaElectrons, self.NOrbitals)
           self.AlphaDoubleExcitations = self.double_excitations(self.NCoreOrbitals, self.NAlphaElectrons, self.NAlphaElectrons, self.NOrbitals)
           self.BetaDoubleExcitations = self.double_excitations(self.NCoreOrbitals, self.NBetaElectrons, self.NBetaElectrons, self.NOrbitals)
        elif 'SF' in self.ExcitationType:
           # First include ground state
           spin_flip_states = [[0,alpha_ground[:],beta_ground[:]]]
           alpha_occupied = alpha_ground[:]; beta_occupied = beta_ground[:]
           # Determine CI level if appropriate
           if 'CISDTQ' in self.ExcitationType: ci_level = 4
           elif 'CISDT' in self.ExcitationType: ci_level = 3
           elif 'CISD' in self.ExcitationType: ci_level = 2
           elif 'CIS' in self.ExcitationType: ci_level = 1
           else: ci_level = 0
           # Do first spin flip
           alpha_occupied[self.NAlphaElectrons] = 1; beta_occupied[self.NBetaElectrons-1] = 0
           # Attach appropriate electronic states and excitation lists specifying determinants
           # Generate spin_flip_states, these are a list with list elemenst where the first element is
           # the multiplicity of the spin flip state used to generate and the next tw elements are 
           # the the alpha and beta occupancies of the resultant (not spin flipped) state 
           if 'SFS' in self.ExcitationType:
              self.States += [(ElectronicState(alpha_occupied, beta_occupied, self.NOrbitals))]
              if 'CI' in self.ExcitationType:
                 spin_flip_states += self.make_CI_states(alpha_ground,beta_ground,self.NAlphaElectrons,self.NBetaElectrons,ci_level=min(ci_level,2),sf_level=1,istate=1)
              else:
                 spin_flip_states += self.make_spinflip_states(alpha_ground,beta_ground,self.NAlphaElectrons,self.NBetaElectrons,level=1,istate=1)
           # Do next spin flip (even if we don't use it)
           alpha_occupied[self.NAlphaElectrons+1] = 1; beta_occupied[self.NBetaElectrons-2] = 0
           # Attach appropriate electronic states and excitation lists specifying determinants
           if 'SFSD' in self.ExcitationType:
              self.States += [(ElectronicState(alpha_occupied, beta_occupied, self.NOrbitals))]
              if 'CI' in self.ExcitationType:
                 spin_flip_states += self.make_CI_states(alpha_ground,beta_ground,self.NAlphaElectrons,self.NBetaElectrons,ci_level=ci_level,sf_level=2,istate=2)
              else:
                 spin_flip_states += self.make_spinflip_states(alpha_ground,beta_ground,self.NAlphaElectrons,self.NBetaElectrons,level=2,istate=2)
           if 'SFD' in self.ExcitationType:
              self.States += [(ElectronicState(alpha_occupied, beta_occupied, self.NOrbitals))]
              if 'CI' in self.ExcitationType:
                 spin_flip_states += self.make_CI_states(alpha_ground,beta_ground,self.NAlphaElectrons,self.NBetaElectrons,ci_level=ci_level,sf_level=2,istate=1)
              else:
                 spin_flip_states += self.make_spinflip_states(alpha_ground,beta_ground,self.NAlphaElectrons,self.NBetaElectrons,level=1,istate=1)
                 spin_flip_states += self.make_spinflip_states(alpha_ground,beta_ground,self.NAlphaElectrons,self.NBetaElectrons,level=2,istate=1)
        else:
           if self.ExcitationType is not None:
              print("Error: excitation type " + self.ExcitationType + " not recognised in set_excitations")
              sys.exit() 

        #-------------------------------------------------------------#
        #   Generate excited states, each an ElectronicState object   #
        #-------------------------------------------------------------#

        ### Generate occupancy lists and ElectronicState instances for excited states ###
        if ('SF' not in self.ExcitationType):

           # Common loop to make excitations and electronic states for all cases except single beta excitations and double excitations
           beta_occupied = beta_ground
           for (i,alpha_excitation) in enumerate(self.AlphaExcitations):
               # Do alpha excitation once and for all up front
               if alpha_excitation != []:
                  alpha_occupied = self.do_excitation(alpha_ground, alpha_excitation)
               # Now also work out what beta excitations to do on a case-by-case basis
               else:
                  if self.ExcitationType == 'CUSTOM-PAIRED':
                     alpha_occupied = alpha_ground
                     beta_occupied = self.do_excitation(beta_ground, self.BetaExcitations[i])
                  else:
                     break
               if (self.ExcitationType == 'S') or (self.ExcitationType == 'SD') or (self.ExcitationType == 'HOMO-LUMO'):
                  beta_occupied = beta_ground
               if self.ExcitationType == 'DOUBLE-PAIRED':
                  beta_occupied = alpha_occupied[:]
               # Generate singly excited states
               self.States += [(ElectronicState(alpha_occupied, beta_occupied, self.NOrbitals))]

           # Do single beta excitations separately
           if (self.ExcitationType == 'S' or self.ExcitationType == 'SD'):
               alpha_occupied = alpha_ground
               for beta_excitation in self.BetaExcitations:
                  if beta_excitation != []:
                     beta_occupied = self.do_excitation(beta_ground, beta_excitation)
                     self.States += [(ElectronicState(alpha_occupied, beta_occupied, self.NOrbitals))]
                  else:
                     break

           # Generate doubly excited states (pairs of singles then single-spin doubles) 
           if (self.ExcitationType == 'D') or self.ExcitationType == 'SD':
              for alpha_excitation in self.AlphaExcitations:
                 alpha_occupied = self.do_excitation(alpha_ground, alpha_excitation)
                 for beta_excitation in self.BetaExcitations:
                    beta_occupied = self.do_excitation(beta_ground, beta_excitation)
                    self.States += [(ElectronicState(alpha_occupied, beta_occupied, self.NOrbitals))]
              for alpha_double_excitation in self.AlphaDoubleExcitations:
                 alpha_occupied = self.do_double_excitation(alpha_ground, alpha_double_excitation)
                 self.States += [(ElectronicState(alpha_occupied, beta_ground, self.NOrbitals))]
              for beta_double_excitation in self.BetaDoubleExcitations:
                 beta_occupied = self.do_double_excitation(beta_ground, beta_double_excitation)
                 self.States += [(ElectronicState(alpha_ground, beta_occupied, self.NOrbitals))]

        ### Store number of electronic states (ground + excited) ###
        self.NStates = len(self.States)
        self.SpinFlipStates = spin_flip_states 

    #=================================================================#
    #              Utility functions for this section                 #
    def single_excitations(self, occ_start, occ_stop, virt_start, virt_stop):
        """Takes the number of electrons of a particular spin and the number
        of orbitals and returns the list of pairs corresponding to all single
        excitations"""
        excitations = []
        for i in range(occ_start,occ_stop):
            for j in range(virt_start,virt_stop):
                excitations.append([i,j])
        return excitations

    def double_excitations(self, occ_start, occ_stop, virt_start, virt_stop):
        """Takes the number of electrons of a particular spin and the number
        of orbitals and returns the list of pairs corresponding to all single
        excitations"""
        excitations = []
        for i in range(occ_start,occ_stop):
            for j in range(i+1,occ_stop): 
                for k in range(virt_start,virt_stop):
                    for l in range(k+1,virt_stop): 
                        excitations.append([i,j,k,l])
        return excitations

    def do_excitation(self, ground_occ, excitation):
        occupied = copy.deepcopy(ground_occ)
        if excitation != []:
            occupied[excitation[0]] = 0
            occupied[excitation[1]] = 1
        else:
            print("""Warning: shouldn't be calling do_excitation without a specified excitation pair""")
        return occupied

    def do_double_excitation(self, ground_occ, excitation):
        occupied = copy.deepcopy(ground_occ)
        if excitation != []:
            occupied[excitation[0]] = 0
            occupied[excitation[1]] = 0
            occupied[excitation[2]] = 1
            occupied[excitation[3]] = 1
        else:
            print("""Warning: shouldn't be calling do_double_excitation without a specified excitation quartet""")
        return occupied

    def make_spinflip_states(self,alpha_occ,beta_occ,n_alpha,n_beta,level,istate):
        beta_occ[n_beta-1] = 0; beta_occ[n_alpha] = 1
        spin_flip_states = []
        if level == 1:
           spin_flip_states.append([level,alpha_occ,beta_occ])
           for i in range(n_beta-1,n_alpha):
              new_alpha = alpha_occ[:]; new_beta = beta_occ[:]
              new_alpha[i] = 0; new_beta[i] = 1
              new_alpha[n_alpha] = 1; new_beta[n_alpha] = 0
              spin_flip_states.append([istate,new_alpha,new_beta])
        if level == 2:
           beta_occ[n_beta-2] = 0; beta_occ[n_alpha+1] = 1
           spin_flip_states.append([level,alpha_occ,beta_occ])
           for i in range(n_beta-2,n_alpha):
             for k in range(n_alpha,n_alpha+2):
                new_alpha = alpha_occ[:]; new_beta = beta_occ[:]
                new_alpha[i] = 0; new_beta[i] = 1
                new_alpha[k] = 1; new_beta[k] = 0
                spin_flip_states.append([level,new_alpha,new_beta])
             for j in range(i,n_alpha):
                new_alpha = alpha_occ[:]; new_beta = beta_occ[:]
                new_alpha[i] = 0; new_alpha[j] = 0; new_beta[i] = 1; new_beta[j] = 1
                new_alpha[n_alpha] = 1; new_alpha[n_alpha+1] = 1; new_beta[n_alpha] = 0; new_beta[n_alpha+1] = 0
                spin_flip_states.append([istate,new_alpha,new_beta])
        return spin_flip_states
         
    def make_CI_states(self,alpha_ground,beta_ground,n_alpha,n_beta,ci_level,sf_level,istate):

        ci_states = []

        # Set up excitation strings and generate excited state occupancy lists for single excitations
        single_excitations = self.single_excitations(n_beta-sf_level,n_beta,n_alpha,n_alpha+sf_level)
        if sf_level == 2 and istate == 2:
           excluded_single_excitations = self.single_excitations(n_beta-1,n_beta,n_alpha,n_alpha+1)
           remaining_single_excitations = [s for s in single_excitations if s not in excluded_single_excitations]
           single_excitations = remaining_single_excitations[:]  
        alpha_singles = []; beta_singles = []
        for single_excitation in single_excitations:
           alpha_singles.append(self.do_excitation(alpha_ground,single_excitation))
           beta_singles.append(self.do_excitation(beta_ground,single_excitation)) 

        # Generate excitation strings and excited state occupancies for double excitations in case they are needed 
        alpha_doubles = []; beta_doubles = []
        if sf_level > 1: 
           double_excitations = self.double_excitations(n_beta-sf_level,n_beta,n_alpha,n_alpha+sf_level)
           for double_excitation in double_excitations:
              alpha_doubles.append(do_double_excitation(alpha_ground,double_excitation))
              beta_doubles.append(do_double_excitation(beta_ground,double_excitation))
              
        # Generate excited state occupancy sets 
        if ci_level > 0:
           for alpha_single in alpha_singles:
              ci_states.append([istate,alpha_single,beta_ground])
           for beta_single in beta_singles:
              ci_states.append([istate,alpha_ground,beta_single])

           if ci_level > 1:
              for alpha_single in alpha_singles:
                 for beta_single in beta_singles:
                    ci_states.append([istate,alpha_single,beta_single])
              for alpha_double in alpha_doubles:
                 ci_states.append([istate,alpha_double,beta_ground]) 
              for beta_double in beta_doubles:
                 ci_states.append([istate,alpha_ground,beta_double])
  
              if ci_level > 2:
                 for alpha_double in alpha_doubles:
                    for beta_single in beta_singles:
                       ci_states.append([istate,alpha_double,beta_single])
                 for alpha_single in alpha_singles:
                    for beta_double in beta_doubles:
                       ci_states.append([istate,alpha_single,beta_double])

                 if ci_level > 3:
                    for alpha_double in alpha_doubles:
                       for beta_double in beta_doubles:
                          ci_states.append([istate,alpha_double,beta_double])

        return ci_states
           

#=====================================================================#
#                      MOLECULE SUBCLASS - ATOM                       #
#=====================================================================#

class Atom:
    def __init__(self,index,row,basis_set,cartesian_l,max_l,to_bohr):
        [label,Z,x,y,z] = row
        self.Index = index
        self.Label = label.upper()
        self.NuclearCharge = Z
        self.Coordinates = [x*to_bohr,y*to_bohr,z*to_bohr]
        self.update_atomic_basis(basis_set,cartesian_l,max_l) 
    def update_atomic_basis(self,basis_set,cartesian_l,max_l):
        self.Basis = []
        self.NFunctions = 0
        self.MaxAng = 0
        basis_data = basis.get[basis_set][self.Label]
        for function in basis_data:
          if function[0] <= max_l:
            cgtf = ContractedGaussian(function,cartesian_l)
            self.Basis.append(cgtf)
            self.NFunctions += cgtf.NAngMom
            if function[0] > self.MaxAng:
                self.MaxAng = function[0]
    def update_coords(self,xyz):
        self.Coordinates = xyz

#---------------------------------------------------------------------#
#                 ATOM SUBCLASS - CONTRACTED GAUSSIAN                 #
#---------------------------------------------------------------------#

class ContractedGaussian:
    def __init__(self,function,cartesian_l):
        self.AngularMomentum = function[0]
        self.NAngMomCart  = c.nAngMomCart[self.AngularMomentum]
        self.NAngMomSpher = c.nAngMomSpher[self.AngularMomentum]
        self.Primitives = function[1:]
        self.NPrimitives = len(self.Primitives)
        self.Exponents = numpy.array([exponent for [exponent,cc] in self.Primitives])
        self.DoubleExponents = numpy.multiply(self.Exponents,2.0)
        self.ScaledCCs = [cc*(2*exponent)**((self.AngularMomentum+1.5)/2.0) for [exponent,cc] in self.Primitives]
        if self.AngularMomentum in cartesian_l:
           self.NAngMom = self.NAngMomCart
           self.CartToSpher = numpy.identity(self.NAngMomCart)
        else:
           self.NAngMom = self.NAngMomSpher
           self.CartToSpher = numpy.array(transform_basis.cart_to_spher[self.AngularMomentum]) 
        angmom_scale_factors = []
        for lx in range(self.AngularMomentum,-1,-1):
          for ly in range(self.AngularMomentum-lx,-1,-1):
            lz = self.AngularMomentum-lx-ly
            glx = math.gamma(lx+0.5); gly = math.gamma(ly+0.5); glz = math.gamma(lz+0.5)
            angmom_scale_factors.append((glx*gly*glz)**-0.5)
        self.ContractionScaling = angmom_scale_factors

#=====================================================================#
#               MOLECULE SUBCLASS - ELECTRONIC STATE                  #
#=====================================================================#

class ElectronicState:
    def __init__(self,alpha_occupancy,beta_occupancy,n_orbitals):
        total_occupancy = [a+b for a,b in zip(alpha_occupancy,beta_occupancy)]
        self.NAlpha = sum(alpha_occupancy)
        self.NBeta = sum(beta_occupancy)
        self.S2 = None
        self.Energy = None
        self.TotalEnergy = None
        self.Total = Matrices(n_orbitals,occupancy=total_occupancy,total=True)
        self.Alpha = Matrices(n_orbitals,occupancy=alpha_occupancy) 
        self.Beta = Matrices(n_orbitals,occupancy=beta_occupancy) 
#        self.Gradient = gradient
#        self.Hessian = hessian
        self.AlphaDIIS = StoreDIIS()
        self.BetaDIIS = StoreDIIS()

#---------------------------------------------------------------------#
#                ELECTRONIC STATE SUBCLASS - MATRICES                 #
#---------------------------------------------------------------------#

class Matrices:
    def __init__(self,n_orbitals,occupancy=[],total=False):
        self.Occupancy = copy.copy(occupancy)
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
       # note that ia_vec & ib_vec are vectors of core and overlap matrix 
       # indices for each spherical polar cgtf of length NAngMomSpher
       self.Centre1 = Shell(coords_a,cgtf_a,ia,ia_vec)
       self.Centre2 = Shell(coords_b,cgtf_b,ib,ib_vec)
       n_alpha = cgtf_a.NPrimitives; alpha_exponents = cgtf_a.Exponents; A = numpy.array(coords_a)
       n_beta = cgtf_b.NPrimitives; beta_exponents = cgtf_b.Exponents; B = numpy.array(coords_b)
       sigmas = numpy.zeros((n_alpha,n_beta),dtype="double")
       overlaps = numpy.zeros((n_alpha,n_beta),dtype="double")
       centres = numpy.zeros((n_alpha,n_beta,3),dtype="double")
#_C_INTS
       _c_ints.shellpair_quantities(sigmas,overlaps,centres,alpha_exponents[:],A[:],n_alpha,beta_exponents[:],B[:],n_beta)
#_C_INTS
       self.Ltot = cgtf_a.AngularMomentum + cgtf_b.AngularMomentum
       self.PrimitivePairSigmas = sigmas
       self.PrimitivePairOverlaps = overlaps
       self.PrimitivePairCentres = centres
       self.PrimitivePairHalfSigmas = numpy.multiply(sigmas,0.5)
       self.CentreDisplacement = numpy.subtract(A,B)
       nmA = numpy.array([cgtf_a.ContractionScaling]); ccA = numpy.array([cgtf_a.ScaledCCs]) 
       nmB = numpy.array([cgtf_b.ContractionScaling]); ccB = numpy.array([cgtf_b.ScaledCCs]) 
       self.Normalization = numpy.array([(nmA.T).dot(nmB).flatten()])
       self.ContractionCoeffs = (ccA.T).dot(ccB)
#       nlsA = cgtf_a.NAngMomSpher; nlcA = cgtf_a.NAngMomCart 
#       nlsB = cgtf_b.NAngMomSpher; nlcB = cgtf_b.NAngMomCart 
       nlA = cgtf_a.NAngMom; nlcA = cgtf_a.NAngMomCart 
       nlB = cgtf_b.NAngMom; nlcB = cgtf_b.NAngMomCart 
       self.BasisTransform = numpy.zeros((nlA*nlB,nlcA*nlcB))
#       self.BasisTransform = numpy.zeros((nlsA*nlsB,nlcA*nlcB))
       ispher = -1
       for sA in cgtf_a.CartToSpher:
         for sB in cgtf_b.CartToSpher:
            ispher += 1; icart = -1
            for cA in sA: 
              for cB in sB:
                icart += 1
                self.BasisTransform[ispher,icart] = cA*cB


#---------------------------------------------------------------------#
#                    SHELLPAIR SUBCLASS - SHELL                       #
#---------------------------------------------------------------------#

class Shell:
    def __init__(self,coords,cgtf,index,index_vec):
       self.Coords = coords
       self.Cgtf = cgtf
       self.Index = index     # Index for the CGTO in atom.Basis
       self.Ivec = index_vec  # Indices of each function within a shell of given angular momentum


#=====================================================================#
#                   GLOBAL UTILITY FUNCTIONS                          #
#=====================================================================#

def remove_punctuation(basis_set):
    basis_set = basis_set.replace('*','s').replace('-','').replace('(','').replace(')','').replace(',','').upper()
    return basis_set

def reorder_MOs(old_MOs, occupancy):
# This function permutes the molecular orbitals for excited states 
# so that the newly occupied orbitals come before newly unoccupied ones
    n_electrons = sum(occupancy)
    new_MOs = old_MOs.copy()
    frm = []                        # list contains indexes of orbitals to be excited from
    to = []                         # list contains indexes of orbitals to be excited to
    for i in range(n_electrons):
        if occupancy[i] == 0:
            frm.append(i)
    for i in range(n_electrons,len(old_MOs[0])):
        if occupancy[i] == 1:
            to.append(i)
    for i in range(len(to)):
        new_MOs[:,[frm[i],to[i]]] = new_MOs[:,[to[i],frm[i]]]
    return new_MOs

def swap_MOs(old_MOs,frm,to):
    new_MOs = old_MOs.copy()
    new_MOs[:,[frm,to]] = new_MOs[:,[to,frm]]
    return new_MOs

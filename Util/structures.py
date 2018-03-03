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
# Import custom-written data modules
from Data import basis
from Data import constants as c
# Import custom-written code modules
from Methods import _c_ints

#=====================================================================#
#  MAIN ROUTINES - called from pychem.py and hartree_fock.py          #
#=====================================================================#
def process_input(section, parser):
    inputs = inputs_return_function(section, parser)
    settings = Settings(inputs, section)
    molecule = Molecule(inputs=inputs, settings=settings)
    return molecule,settings

def inputs_return_function(section, parser):
    def inputs(var_name):
        var = parser.get(section,var_name)
        var = ast.literal_eval(var)
        if isinstance(var, str):
            var = var.upper()
        return var
    return inputs

def update_basis(basis_set):
    molecule = Molecule(basis=basis_set)
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
                mo_read_basis = remove_punctuation(inputs("MO_Read_Basis"))
                self.AlphaMOFile = mo_read_basis + '_' + mo_read_state + '.alpha_MOs' 
                self.BetaMOFile = mo_read_basis + '_' + mo_read_state + '.beta_MOs' 
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

        #----------------------- Excitations --------------------------#
        available_excitation_types = [None, 'SINGLE', 'DOUBLE', 'HOMO-LUMO', 'DOUBLE-PAIRED', 'CUSTOM-SINGLE','CUSTOM-PAIRED',
                                      'S','D','SPIN-FLIP-SINGLE','SFS','SF-S','SPIN-FLIP-DOUBLE','SFD','SF-D']

        try:
           excitation_type = inputs("Excitations").upper()
           if (excitation_type == 'S'): excitation_type = 'SINGLE'
           if (excitation_type == 'D'): excitation_type = 'DOUBLE'
           if (excitation_type == 'SFS') or (excitation_type == 'SF-S'): excitation_type = 'SPIN-FLIP-SINGLE'
           if (excitation_type == 'SFD') or (excitation_type == 'SF-D'): excitation_type = 'SPIN-FLIP-DOUBLE'
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
                 print("""Error: Alpha_Excitations and Beta_Excitations lists must be equal in length if both keywords present,
                          blank sub-lists can be used as placeholders""")
                 sys.exit()
           else:
              excitation_type = 'CUSTOM-SINGLE'

        ### Check that excitations were given in the correct format ###
        try:
           assert excitation_type in available_excitation_types
        except:
           print("""Error: Excitations can be specified by keyword as Excitations = 'Single'/'S', 'Double'/'D', 'Homo-Lumo', 'Double-Paired', 
                    'Spin-Flip-Single'/'SF-S', 'Spin-Flip-Double'/'SF-D'""")
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
        n_electrons = 0
        n_orbitals = 0
        n_core_orbitals = 0
        n_cgtf = 0
        for index,row in enumerate(self.Coords):
            ### Add Atom to Molecule ###
            atom = Atom(index,row,self.Basis,self.CoordsScaleFactor)
            self.Atoms.append(atom)
            n_electrons += c.nElectrons[atom.Label]
            n_core_orbitals += c.nCoreOrbitals[atom.Label]
            n_orbitals += atom.NFunctions
            n_cgtf += len(atom.Basis) 

        #------------ Ground State Electronic Configuration -----------#
        n_electrons = n_electrons - self.Charge
        try:
            n_alpha = (n_electrons + (self.Multiplicity-1))/2
            n_beta  = (n_electrons - (self.Multiplicity-1))/2
        except:
            print('Error: charge and multiplicity inconsistent with specified molecule')
            sys.exit()
        self.NElectrons = n_electrons
        self.NAlphaElectrons = n_alpha
        self.NBetaElectrons = n_beta
        self.NOrbitals = n_orbitals
        self.NCoreOrbitals = n_core_orbitals
        self.NAlphaOrbitals = int(math.ceil(self.NAlphaElectrons/2.0))
        self.NBetaOrbitals = int(math.ceil(self.NBetaElectrons/2.0))
        self.NCgtf = n_cgtf

        #----------- SCF structures - common to all states ------------#
        #           contains ShellPair -> Shell subclasses             #    
        #--------------------------------------------------------------#
        self.Core = numpy.zeros((self.NOrbitals,) * 2)
        self.Overlap = numpy.zeros((self.NOrbitals,) * 2)
        self.NuclearRepulsion = None
        self.X = []
        self.Xt = []
        self.S = []
        self.CoulombIntegrals = numpy.zeros((self.NOrbitals,) * 4) 
        self.ExchangeIntegrals = numpy.zeros((self.NOrbitals,) * 4) 
        ### Generate and store ShellPair data for all unique pairs of CGTFs ###
        shell_pairs = {}
        ia = -1; ia_count = 0;
        for atom_a in self.Atoms:
           for cgtf_a in atom_a.Basis:
              ia_vec = [(ia_count + i) for i in range(0,cgtf_a.NAngMom)]   # vector containing the indices each angular momentum function on the atom
              ia += 1; ia_count += cgtf_a.NAngMom                          
              ib_count = 0; ib = -1
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

        if (self.ExcitationType == 'SINGLE') or (self.ExcitationType == 'DOUBLE'):
           self.AlphaExcitations = self.single_excitations(self.NCoreOrbitals, self.NAlphaElectrons, self.NAlphaElectrons, self.NOrbitals)
           self.BetaExcitations = self.single_excitations(self.NCoreOrbitals, self.NBetaElectrons, self.NBetaElectrons, self.NOrbitals)
        if (self.ExcitationType == 'DOUBLE-PAIRED'):
           self.AlphaExcitations = self.single_excitations(self.NCoreOrbitals, self.NBetaElectrons, self.NAlphaElectrons, self.NOrbitals)
        if (self.ExcitationType == 'HOMO-LUMO'):
           self.AlphaExcitations = self.single_excitations(self.NAlphaElectrons-1, self.NAlphaElectrons, self.NAlphaElectrons, self.NAlphaElectrons+1)

        #-------------------------------------------------------------#
        #   Generate excited states, each an ElectronicState object   #
        #-------------------------------------------------------------#

        ### Make occupancy lists for the ground state ###
        alpha_occupancy = [1 for i in range(0,self.NAlphaElectrons)] + [0 for i in range(0,self.NOrbitals-self.NAlphaElectrons)]
        beta_occupancy = [1 for i in range(0,self.NBetaElectrons)] + [0 for i in range(0,self.NOrbitals-self.NBetaElectrons)]

        ### Instantiate ElectronicState for ground state ###
        self.States = [ElectronicState(alpha_occupancy,beta_occupancy,self.NOrbitals)]

        ### Generate occupancy lists and ElectronicState instances for excited states ###
        spin_flip_states = None
        if (self.ExcitationType is not None):

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

           # Or set up spin-flip states
           if 'SPIN-FLIP' in self.ExcitationType:
               spin_flip_states = [[0,alpha_ground[:],beta_ground[:]]]
               alpha_occupied = alpha_ground[:]; beta_occupied = beta_ground[:]
               alpha_occupied[self.NAlphaElectrons] = 1; beta_occupied[self.NBetaElectrons-1] = 0
               self.States += [(ElectronicState(alpha_occupied, beta_occupied, self.NOrbitals))]
               spin_flip_states += self.make_spinflip_states(alpha_ground,beta_ground,self.NAlphaElectrons,self.NBetaElectrons,level=1)
               if 'DOUBLE' in self.ExcitationType:
                  alpha_occupied[self.NAlphaElectrons+1] = 1; beta_occupied[self.NBetaElectrons-2] = 0
                  self.States += [(ElectronicState(alpha_occupied, beta_occupied, self.NOrbitals))]
                  spin_flip_states += self.make_spinflip_states(alpha_ground,beta_ground,self.NAlphaElectrons,self.NBetaElectrons,level=2)

        ### Store number of electronic states (ground + excited) ###
        self.NStates = len(self.States)
        self.SpinFlip = spin_flip_states 

    #=================================================================#
    #              Utility functions for this section                 #
    def single_excitations(self, n_electrons, n_orbitals):
        """Takes the number of electrons of a particular spin and the number
        of orbitals and returns the list of pairs corresponding to all single
        excitations"""
        excitations = []
        n_virtual_orbitals = n_orbitals - n_electrons
        for i in range(1,n_electrons+1):
            for j in range(1,n_virtual_orbitals+1):
                excitations.append([-i,j])
        return excitations

    def do_excitation(self, ground_occ, excitation):
        occupied = copy.deepcopy(ground_occ)
        if excitation != []:
            occupied[excitation[0]] = 0
            occupied[excitation[1]] = 1
        else:
            print("""Warning: shouldn't be calling do_excitation without a specified excitation pair""")
        return occupied

    def make_spinflip_states(self,alpha_occ,beta_occ,n_alpha,n_beta,level):
        beta_occ[n_beta-1] = 0; beta_occ[n_alpha] = 1
        spin_flip_states = []
        if level == 1:
           spin_flip_states.append([level,alpha_occ,beta_occ])
           for i in range(n_beta-1,n_alpha):
              new_alpha = alpha_occ[:]; new_beta = beta_occ[:]
              new_alpha[i] = 0; new_beta[i] = 1
              new_alpha[n_alpha] = 1; new_beta[n_alpha] = 0
              spin_flip_states.append([level,new_alpha,new_beta])
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
                spin_flip_states.append([level,new_alpha,new_beta])
        return spin_flip_states
         
#=====================================================================#
#                      MOLECULE SUBCLASS - ATOM                       #
#=====================================================================#

class Atom:
    def __init__(self,index,row,basis_set,to_bohr):
        [label,Z,x,y,z] = row
        self.Index = index
        self.Label = label.upper()
        self.NuclearCharge = Z
        self.Coordinates = [x*to_bohr,y*to_bohr,z*to_bohr]
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
        self.Exponents = numpy.array([exponent for [exponent,cc] in self.Primitives])
        self.DoubleExponents = numpy.multiply(self.Exponents,2.0)
        self.ScaledCCs = [cc*(2*exponent)**((self.AngularMomentum+1.5)/2.0) for [exponent,cc] in self.Primitives]
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
        self.Occupancy = occupancy
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
       n_alpha = cgtf_a.NPrimitives; alpha_exponents = cgtf_a.Exponents; A = numpy.array(coords_a)
       n_beta = cgtf_b.NPrimitives; beta_exponents = cgtf_b.Exponents; B = numpy.array(coords_b)
       sigmas = numpy.zeros((n_alpha,n_beta),dtype="double")
       overlaps = numpy.zeros((n_alpha,n_beta),dtype="double")
       centres = numpy.zeros((n_alpha,n_beta,3),dtype="double")
#_C_INTS
       _c_ints.shellpair_quantities(sigmas,overlaps,centres,alpha_exponents,A,n_alpha,beta_exponents,B,n_beta)
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

#---------------------------------------------------------------------#
#                    SHELLPAIR SUBCLASS - SHELL                       #
#---------------------------------------------------------------------#

class Shell:
    def __init__(self,coords,cgtf,index,index_vec):
       self.Coords = coords
       self.Cgtf = cgtf
       self.Index = index     # Index for the CGTO in atom.Basis
       self.Ivec = index_vec  # Indexs of the angular momentum functions in a list of all angular momentum functions on the atom


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
    for i in range(n_electrons,len(occupancy)):
        if occupancy[i] == 1:
            to.append(i)
    for i in range(len(to)):
        new_MOs[:,[frm[i],to[i]]] = new_MOs[:,[to[i],frm[i]]]
    return new_MOs


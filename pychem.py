#!/usr/bin/python

# System libraries
import os
import sys
if sys.version_info.major is 2:
   import ConfigParser
else:
   import configparser as ConfigParser
# Custom-written object modules (data and derived data at this level)
from Util import structures
# Custom-written code modules
from Methods import hartree_fock
from Methods import basis_fit
from Methods import mp2
from Methods import properties
from Methods import noci

#======================================================================#
#                           THE MAIN PROGRAM                           #
#======================================================================#
# Process:                                                             # 
# -------------------------------------------------------------------- #
# __main__: loop over sections in input file, store system-independent #
#           data in Settings, and system-dependent data in Molecule    #  
#           then do_calculation                                        #
# -------------------------------------------------------------------- #
# do_calculation:                                                      # 
#   - do HF in initial (usually minimal) basis set, store MOs          #
#   - do MOM-HF for excited states in initial basis set, store MOs     # 
#   for each additional basis set:                                     #
#     - construct guess MOs for all states by basis fitting from       #       
#       initial (minimal) basis set results                            #
#     - perform MOM-HF calculations for all states                     #  
#======================================================================#

def do_calculation(settings, molecule):

    # Open output file, print header
    if os.path.exists(settings.OutFileName): os.remove(settings.OutFileName) 
    settings.OutFile = open(settings.OutFileName,'a+')

    if settings.Method is not None:

        # Do ground state calculation in the starting basis, storing MOs (and 2e ints as appropriate) as we go
        basis_set = settings.BasisSets[0]
        hartree_fock.do(settings, molecule, basis_set)
 
        #-------------------------------------------------------------------
        # Generate starting orbital sets for each of the requested excited states and do calculation in first basis
        for index, state in enumerate(molecule.States[1:], start = 1):
        
            # Reorder MOs if changing orbital occupancy, but not if changing spin multiplicity
            if molecule.NAlphaElectrons == state.NAlpha:
                state.Alpha.MOs = structures.reorder_MOs(molecule.States[0].Alpha.MOs, state.Alpha.Occupancy)
                state.Beta.MOs = structures.reorder_MOs(molecule.States[0].Beta.MOs, state.Beta.Occupancy)

            hartree_fock.do(settings, molecule, basis_set, index)

        #-------------------------------------------------------------------
        # Do larger basis calculations, using basis fitting to obtain initial MOs
        for basis_set in settings.BasisSets[1:]:
        
            # Iterate over list and perform basis fitting on each state, replacing old MOs with new ones 
            alpha_MOs = []; beta_MOs = []
            for state in molecule.States:
                alpha_MOs.append(basis_fit.do(molecule, state.Alpha.MOs, basis_set))
                beta_MOs.append(basis_fit.do(molecule, state.Beta.MOs, basis_set))

            # Make new molecule with a new basis set and copy in old orbitals (fit using new basis set)
            molecule = molecule.update_basis(basis_set)
            for state in molecule.States:
                state.Alpha.MOs = alpha_MOs.pop(0)
                state.Beta.MOs = beta_MOs.pop(0)

            # Iterate over the list of states doing calculations (enforce orthogonality for MOM but not SF-NOCI)
            for index in range(molecule.NStates):
                hartree_fock.do(settings, molecule, basis_set, index)

    #-------------------------------------------------------------------
    # Do post-HF (MP2) calculations in final basis for all single-reference electronic states
    if settings.Method == 'MP2':
        
        mp2.do(settings, molecule) 

    #-------------------------------------------------------------------
    # Do NOCI calculations in final basis, setting up spin-flip basis states
    if settings.Method == 'NOCI':

        if molecule.SpinFlipStates != []:
            spin_flip_states = []
            for [index,alpha_occupancy,beta_occupancy] in molecule.SpinFlipStates:
                state = structures.ElectronicState(alpha_occupancy, beta_occupancy, molecule.NOrbitals)
                state.Alpha.MOs = structures.reorder_MOs(molecule.States[index].Alpha.MOs, alpha_occupancy)
                state.Beta.MOs = structures.reorder_MOs(molecule.States[index].Beta.MOs, beta_occupancy)
                state.TotalEnergy = molecule.States[index].TotalEnergy
                hartree_fock.make_density_matrices(molecule,state)
                spin_flip_states.append(state)
            molecule.States = spin_flip_states

        noci.do(settings, molecule)

    #-------------------------------------------------------------------
    # Compute properties if requested - only mean-field for now
    if settings.JobType == 'PROPERTY':

        properties.calculate(settings, molecule)

    # Close output file
    settings.OutFile.close()

#======================================================================#
# __main__: Process input file, loop over sections, set up data        #
#           structures and call do_calculation                         #
#======================================================================#

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please give an input file")
    else:
        parser = ConfigParser.SafeConfigParser()
        has_read_data = parser.read(sys.argv[1])
        if not has_read_data:
            print("Could not open input file, check you have typed the name correctly")
            sys.exit()
        if len(parser.sections()) == 0:
            print("Input file has no recognisable section headings, format [section_heading]")
            sys.exit() 
        for section in parser.sections():
            molecule,settings = structures.process_input(section, parser) 
            do_calculation(settings, molecule)

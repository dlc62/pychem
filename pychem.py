#!/usr/bin/python

# System libraries
from __future__ import print_function
import sys
if sys.version_info.major is 2:
  import ConfigParser   
else:
  import configparser as ConfigParser 
# Custom-written object modules (data and derived data at this level)
import inputs_structures
# Custom-written code modules
import hartree_fock
import basis_fit
import util
import printf
import mp2

#======================================================================#
#                           THE MAIN PROGRAM                           #
#                  Compatible with python2.7 or higher                 #
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

    # Open output file
    printf.initialize(settings)

    # Do ground state calculation in the starting basis, storing MOs (and 2e ints as appropriate) as we go
    hartree_fock.do(settings, molecule)

    #-------------------------------------------------------------------
    # Generate starting orbital sets for each of the requested excited states and do calculation in first basis
    for index, state in enumerate(molecule.States[1:], start = 1):
        
        state.add_MOs(util.excite(molecule.States[0].alpha_MOs, state.AlphaOccupancy, molecule.NAlphaElectrons),
                      util.excite(molecule.States[0].beta_MOs, state.BetaOccupancy, molecule.NBetaElectrons))

        hartree_fock.do(settings, molecule, state, index)

    # Dump MOs to file for initial basis set, all states
    util.store('MOs', molecule.States, settings.SectionName, settings.BasisSets[0])

    for basis_set in settings.BasisSets[1:]:
        
        # Iterate over list and perform basis fitting on each state, replacing old MOs with new ones 
        alpha_MOs = []; beta_MOs = []
        for state in molecule.States:
            alpha_MOs.append(basis_fit.do(molecule, state.alphaMOs, basis_set))
            beta_MOs.append(basis_fit.do(molecule, state.betaMOs, basis_set))

        # Make new molecule with a new basis set and copy in old orbitals (fit using new basis set)
        molecule = inputs_structures.new_molecule(basis_set)
        for state in molecule.States:
            state.Alpha.MOs = alpha_MOs.pop(0)
            state.Beta.MOs = beta_MOs.pop(0)

        # Iterate over the list of states doing calculations while enforcing orthogonality
        for index, state in enumerate(molecule.States):
            hartree_fock.do(settings, molecule, state, index)

        # Dump MOs to file for other basis sets, all states 
        util.store('MOs', molecule.States, settings.SectionName, basis_set)
    #-------------------------------------------------------------------

    # Do state-specific MP2 in final basis only
    if settings.Method == 'MP2':
        for index, state in enumerate(molecule.States):
            mp2.do(settings, molecule, state, index) 

    # Close output file
    printf.finalize(settings)

#======================================================================#
# __main__: Process input file, loop over sections, set up data        #
#           inputs_structures and call do_calculation                  #
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
            molecule,settings = inputs_structures.process_input(section, parser) 
            settings.set_outfile(section)
            do_calculation(settings, molecule)

#!/usr/bin/env python

# Hard coding the interpreter path leads to all kinds of unintuitize bahaviour 
# including breaking virtualenvs 

# System libraries
import os
import sys
if sys.version_info.major is 2:
   import ConfigParser
else:
   import configparser as ConfigParser
# Custom-written object modules (data and derived data at this level)
from Util import structures, printf 
# Custom-written code modulemport ipdb; ipdb.set_trace() s
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

def print_commit(outfile):
    try:
        git_dir = os.path.dirname(os.path.realpath(__file__)) + "/.git/"
        contents = "ref: HEAD"

        while contents.startswith("ref: "):
            ref = contents.split()[1]
            with open(git_dir + ref) as f:
                contents = f.read().strip()

        printf.text(outfile, "pychem git commit hash: " + contents)
    except:
        printf.text(outfile, "Could not find pychem git hash")

def do_calculation(settings, molecule):

    # Open output file, print header
    if os.path.exists(settings.OutFileName): os.remove(settings.OutFileName) 
    settings.OutFile = open(settings.OutFileName,'a+')

    print_commit(settings.OutFile)

    # Calculate HF states from highest spin multiplicity to lowest
    state_order = [index for index in range(0,molecule.NStates)]
    if molecule.ExcitationType is not None:
        if 'SF' in molecule.ExcitationType: state_order.reverse()

    if settings.Method is not None:

        # Do ground state calculation in the starting basis, storing MOs (and 2e ints as appropriate) as we go
        basis_set = settings.BasisSets[0]
        hartree_fock.do(settings, molecule, basis_set, state_order[0])
 
        #-------------------------------------------------------------------
        # Generate starting orbital sets for each of the requested excited states and do calculation in first basis
        for i in range(1,molecule.NStates):
            index = state_order[i]; prev_index = state_order[i-1]
            state = molecule.States[index]; prev_state = molecule.States[prev_index]

            # Use higher spin-multiplicity virtuals as starting guess for spin-broken UHF beta orbitals
            if 'SF' in molecule.ExcitationType and state.Alpha.Occupancy != state.Beta.Occupancy:
                state.Alpha.MOs = prev_state.Alpha.MOs[:,:]
                state.Beta.MOs = prev_state.Alpha.MOs[:,:]
                keep_constrained = (settings.SCF.ConstrainExcited and index != 0)
                if settings.SCF.Reference == "UHF" and not keep_constrained:
                    n_swap = prev_state.NAlpha - state.NAlpha
                    for i_swap in range(0,n_swap):
                        to = state.NBeta - 1 + i_swap
                        frm = prev_state.NAlpha - 1 - i_swap
                        state.Beta.MOs = structures.swap_MOs(state.Beta.MOs, to, frm)
            # Reorder MOs if changing orbital occupancy, but not if changing spin multiplicity
            elif molecule.NAlphaElectrons == state.NAlpha:
                state.Alpha.MOs = structures.reorder_MOs(molecule.States[prev_index].Alpha.MOs, state.Alpha.Occupancy)
                state.Beta.MOs = structures.reorder_MOs(molecule.States[prev_index].Beta.MOs, state.Beta.Occupancy)

            hartree_fock.do(settings, molecule, basis_set, index, initial_run=False)

        #-------------------------------------------------------------------
        # Do larger basis calculations, using basis fitting to obtain initial MOs
        for basis_set in settings.BasisSets[1:]:
            # Iterate over list and perform basis fitting on each state, replacing old MOs with new ones 
            alpha_MOs = []; beta_MOs = []
            for state in molecule.States:
                alpha_MOs.append(basis_fit.do(molecule, state.Alpha.MOs, basis_set))
                beta_MOs.append(basis_fit.do(molecule, state.Beta.MOs, basis_set))

            # Update basis set for this molecule by copying in old orbitals (fit using new basis set)
            # Except the first one - just use the core guess for the ground state
            molecule = structures.update_basis(molecule, basis_set)
            for index, state in enumerate(molecule.States[1:]):
                state.Alpha.MOs = alpha_MOs[index+1]
                state.Beta.MOs = beta_MOs[index+1]
                state.Energy = 0.0

            # Iterate over the list of states doing calculations (enforce orthogonality for MOM but not SF-NOCI)
            for index in range(molecule.NStates):
                initial = index == 0
                hartree_fock.do(settings, molecule, basis_set, index, initial_run=initial)

    #-------------------------------------------------------------------
    # Do post-HF (MP2) calculations in final basis for all single-reference electronic states
    if settings.Method.startswith('MP2'):
        
        mp2.do(settings, molecule) 

    #-------------------------------------------------------------------
    # Do NOCI calculations in final basis, setting up spin-flip basis states
    if settings.Method.startswith('NOCI'):

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


def main(input_file):
    parser = ConfigParser.SafeConfigParser()
    try:
        has_read_data = parser.read(input_file)
    except ConfigParser.MissingSectionHeaderError:
        print("Input file has no recognisable section headings, format: [section_heading]")
        sys.exit() 

    if len(has_read_data) == 0:
        print("Could not open input file, check you have typed the name correctly")
        sys.exit()
    for section in parser.sections():
        molecule,settings = structures.process_input(section, parser) 
        do_calculation(settings, molecule)
    return molecule

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please give an input file")
    else:
        main(sys.argv[1])

# System libraries
from __future__ import print_function
import os
import numpy
# Custom-written data modules
import Data.constants as c

# pretty printing to the terminal for numpy arrays
numpy.set_printoptions(precision=5, linewidth=300)

def initialize(settings, method = None):
    if method == None:
        try:
           os.remove(settings.OutFileName)
        except:
           pass
        settings.OutFile = open(settings.OutFileName,'a')

def finalize(settings, method = None):
    settings.OutFile.close()

def print_to_file(outfile,string):
    try:
       outfile.write(string)
    except:
       print(string)

#-----------------------------------------------------------------

def HF_Initial(molecule, this, settings):

    if settings.PrintLevel != 'MINIMAL':
        outString = ""
        outString += '****************************************************' + '\n'
        outString +=  ' Initialization ' + '\n'
        outString += '****************************************************' + '\n\n'
        outString += 'Nuclear repulsion energy ' + str(molecule.NuclearRepulsion) + '\n\n'
        outString += 'Core Fock matrix' + '\n'
        outString +=  str(molecule.Core) + '\n\n'
        outString +=  'Guess alpha density matrix' + '\n'
        outString +=  str(this.Alpha.Density)  + '\n\n'
        if numpy.all(this.Alpha.Density != this.Beta.Density):
            outString +=  'Guess beta density matrix' + '\n'
            outString +=  str(this.Beta.Density)  + '\n\n'
        outString +=  '****************************************************' + '\n'
        outString +=  ' Hartree-Fock iterations ' + '\n'
        outString +=  '****************************************************' + '\n\n'

        print_to_file(settings.OutFile, outString)

        if settings.PrintToTerminal:
            print(outString)

#Possibly need to allow this to print the coulomb and exchange matrices (if PrintLevel == VERBOSE)
def HF_Loop(this, settings, cycles, dE, diis_error, final):

    #testing to see if the alpha and beta orbital energies are the same
    #equalites = map( (lambda x,y: x == y), alpha_energies, beta_energies)
    #restricted = reduce( (lambda x,y: x and y), equalites, True)
    restricted = numpy.all(this.Alpha.Energies == this.Beta.Energies)

    outString = ''
    outString += "Cycle: " + str(cycles) + '\n'
    outString += "Total Energy: " + str(this.TotalEnergy) + '\n'
    outString += "Change in energy: " + str(dE) + '\n'
    if diis_error != None:                 #stops this from printing when DIIS is disabled
        outString += "DIIS Error: " + str(diis_error) + '\n'

    if (settings.PrintLevel == 'DEBUG') or (final is True):
        outString += "Alpha Orbital Energies" + '\n'
        outString += str(this.Alpha.Energies) + '\n'
        #Find a better way to do this comparison
        if restricted == False:
            outString += "Beta Orbital Energies" + '\n'
            outString += str(this.Beta.Energies) + '\n'

    if (settings.PrintLevel == 'DEBUG') or (final is True):
        outString += "Alpha MOs" + '\n'
        outString += str(this.Alpha.MOs) + '\n'
        if restricted == False:
            outString += "Beta MOs" + '\n'
            outString += str(this.Beta.MOs) + '\n'

    if (settings.PrintLevel == 'DEBUG'):
        outString += "Alpha Density Matrix" + '\n'
        outString += str(this.Alpha.Density) + '\n'
        outString += "Alpha Fock Matrix" + '\n'
        outString += str(this.Alpha.Fock) + '\n'
        if restricted == False:
            outString += "Beta Density Matrix" + '\n'
            outString += str(this.Beta.Density) + '\n'
            outString += "Beta Fock Matrix" + '\n'
            outString += str(this.Beta.Fock) + '\n'

    if final and this.S2 != None:
        outString += "<S^2> = %.2f\n" % this.S2
        outString += "Alpha Occupany: {}\n".format(this.AlphaOccupancy)
        outString += "Beta Occupancy: {}\n".format(this.BetaOccupancy)

    outString += '----------------------------------------------------' + '\n'

    if settings.PrintToTerminal:
        print(outString)
    print_to_file(settings.OutFile, outString)

def HF_Final(settings):
    outString =  '                       End                          ' + '\n'
    outString += '----------------------------------------------------' + '\n'
    print_to_file(settings.OutFile, outString)

def MP2_Final(settings, state_index, EMP2):
    outString =  ' Total MP2 energy for state ' + str(state_index) + ' = ' + str(EMP2) + '\n'
    outString += '----------------------------------------------------' + '\n'
    outString +=  '                 End (No, really!)                  ' + '\n'
    outString += '----------------------------------------------------' + '\n'
    print_to_file(settings.OutFile, outString)

#    def MOM(self, alpha_overlaps, beta_overlaps):
#        outString = "Alpha Overlap Vector" + '\n'
#        outString += str(alpha_overlaps) + '\n'
#        outString += "Beta Overlap Vector" + '\n'
#        outString += str(beta_overlaps)
#        print_to_file(self.outFile, outString)

def NOCI(settings, hamil, overlaps, states, energies):

    outString = "\n=== NOCI Output ===\n"

    if settings.NOCI.print_level > 1:
        outString += "Hamiltonian\n{}\n".format(hamil)
    if settings.NOCI.print_level > 2:
        outString += "State Overlaps\n{}\n".format(overlaps)

    outString += "States\n{}\n".format(states)
    outString += "State Energies\n{}\n".format(energies)
    print_to_file(settings.OutFile, outString)

    if settings.PrintToTerminal:
        print(outString)

def printf(settings, to_print):
    print_to_file(settings, to_print)
    if settings.PrintToTerminal:
        print(to_print)

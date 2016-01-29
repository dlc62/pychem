
# need new functionally and removed the need of having hartree_fock.py inclding 
# literal strings for printing 
import input
import numpy

#may need to import input for this to work 
class PrintSettings:
    def __init__(self):
        try:
            self.SCFPrint = input.SCFPrint 
        except:
            self.SCFPrint = 1 
        try:
            self.SCFFinalPrint = input.SCFFinalPrint
        except: 
            self.SCFFinalPrint = 2
        try:
            self.DIISPrint = input.DIISPrint 
        except:
            self.DIISPrint = 0 
        try: 
            self.MinimalPrint = input.minimalPrint 
        except:
            self.MinimalPrint = False
        try:
            self.MOMPrint = input.MOMPrint 
        except:
            self.MOMPrint = 0
        try:
            self.OutFile = input.OutFile 
        except:
            self.OutFile = ''
        
    def finalPrint(self):
        outString =  '                       End                          ' + '\n'
        outString += '----------------------------------------------------' + '\n'
        self.outPrint(outString)
        if self.OutFile != '':
            self.newFile.close()

    def outPrint(self,string):
        if self.OutFile == '':
            print string 
        else:
            try:
                self.newFile.write(string)
                self.newFile.write('\n')
            except:
                print "Cannot write to file: printing output to terminal"
                print string
                self.OutFile = ''

# Currnely these functions just replicate the existing behaviour at the default 
# input values, will need to find an elegant way of customising output in 
# the absence of a switch/case construct 

    def PrintInitial(self,nuclearRepulsion, coreFock, densities):
        if self.OutFile != '':
            print "Calculation Running"
            self.newFile = open(self.OutFile, 'a')

        if self.MinimalPrint != True:
            outString = ""
            outString += '****************************************************' + '\n'
            outString +=  ' Initialization ' + '\n'
            outString += '****************************************************' + '\n'
            outString += 'Nuclear repulsion energy ' + str(nuclearRepulsion) + '\n'
            outString +=  'Core Fock matrix' + '\n'
            outString +=  str(coreFock) + '\n'

            outString +=  'Guess alpha density matrix' + '\n'
            outString +=  str(densities.alpha)  + '\n'
            if numpy.all(densities.alpha != densities.beta):
                outString +=  'Guess beta density matrix' + '\n'
                outString +=  str(densities.beta)  + '\n'

            outString +=  '****************************************************' + '\n'
            outString +=  ' Hartree-Fock iterations ' + '\n'
            outString +=  '****************************************************' + '\n'

            self.outPrint(outString)

    def PrintMOM(self):
        return 0
    
    def PrintDIIS(self): 
        return 0  

    #Possibly need to allow this to print the coloumb and exhange matrices 
    def PrintLoop(self, cycles, alpha_energies, beta_energies, densities,
                   focks, alpha_MOs, beta_MOs, dE, energy, DIIS_error):
    
        #testing to see if the alpha and beta orbital energies are the same
        #equalites = map( (lambda x,y: x == y), alpha_energies, beta_energies)
        #restricted = reduce( (lambda x,y: x and y), equalites, True)
        restricted = numpy.all(alpha_energies == beta_energies)

        outString = ''
        outString += "Cycle: " + str(cycles) + '\n'
        outString += "Total  Energy: " + str(energy) + '\n'
        outString += "Change in energy: " + str(dE) + '\n'
        if DIIS_error != 0:                 #stops this from printing when DIIS is dissabled 
            outString += "DIIS Error: " + str(DIIS_error) + '\n'

        if self.SCFPrint > 0:
            outString += "Alpha Orbial Energies" + '\n'
            outString += str(alpha_energies) + '\n'
            #Find a better way to do this comparison
            if restricted == False:  
                outString += "Beta Orbital Energies" + '\n'
                outString += str(beta_energies) + '\n'

        if self.SCFPrint > 1:
            outString += "Alpha MOs" + '\n'
            outString += str(alpha_MOs) + '\n'
            if restricted == False: 
                outString += "Beta MOs" + '\n'
                outString += str(beta_MOs) + '\n'

        if self.SCFPrint > 2:
            outString += "Alpha Density Matrix" + '\n'
            outString += str(densities.alpha) + '\n'
            outString += "Alpha Fock Matrix" + '\n'
            outString += str(focks.alpha) + '\n'
            if restricted == False: 
                outString += "Beta Density Matrix" + '\n'
                outString += str(beta_density_matrix) + '\n'
                outString += "Beta Fock Matrix" + '\n'
                outString += str(focks.beta) + '\n'
        outString += '----------------------------------------------------'

        self.outPrint(outString)

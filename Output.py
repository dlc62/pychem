# Object for handeling printing output from code can be easily extended to include 
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
            self.minimalPrint = input.minimalPrint 
        except:
            self.minimalPrint = False
        try:
            self.MOMPrint = input.MOMPrint 
        except:
            self.MOMPrint = 0
        

# Currnely these functions just replicate the existing behaviour at the default 
# input values, will need to find an elegant way of customising output in 
# the absence of a switch/case construct 

    def PrintInitial(self,nuclearRepulsion, coreFock, densities):
        if self.minimalPrint != True:
                print '****************************************************'
                print ' Initialization '
                print '****************************************************'
                print 'Nuclear repulsion energy', nuclearRepulsion
                print 'Core Fock matrix'
                print coreFock

                print 'Guess alpha density matrix'
                print densities.alpha 
                if numpy.all(densities.alpha != densities.beta):
                    print 'Guess beta density matrix'
                    print densities.beta 

                print '****************************************************'
                print ' Hartree-Fock iterations '
                print '****************************************************'

    def PrintMOM(self):
        return 0
    
    def PrintDIIS(self): 
        return 0  

    #Possibly need to allow this to print the coloumb and exhange matrices 
    def PrintLoop(self, cycles, alpha_energies, beta_energies, densities,
                   focks, alpha_MOs, beta_MOs, dE, energy, DIIS_error):
    
        #testing to see if the alpha and beta orbital energies are the same
        equalites = map( (lambda x,y: x == y), alpha_energies, beta_energies)
        restricted = reduce( (lambda x,y: x and y), equalites, True)

        print "Cycle: " , cycles 
        print "Total  Energy: " , energy 
        print "Change in energy: " , dE
        if DIIS_error != 0:                 #stops this from printing when DIIS is dissabled 
            print "DIIS Error: " , DIIS_error

        if self.SCFPrint > 0:
            print "Alpha Orbial Energies"
            print alpha_energies
            #Find a better way to do this comparison
            if restricted == False:  
                print "Beta Orbital Energies"
                print beta_energies

        if self.SCFPrint > 1:
            print "Alpha MOs"
            print alpha_MOs
            if restricted == False: 
                print "Beta MOs"
                print beta_MOs 

        if self.SCFPrint > 2:
            print "Alpha Density Matrix"
            print density.alpha
            print focks.alpha
            if restricted == False: 
                print "Beta Density Matrix"
                print beta_density_matrix 
                print "Beta Fock Matrix"
                print focks.beta 
        print '----------------------------------------------------'

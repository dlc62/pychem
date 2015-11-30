# This program takes the density matrix at the end of a calculation 
# and saves it to the SAD_orbitals file for use in latter calculations.
import hartree_fock as HF
import numpy 
from SAD_orbitals import SADget

def getMatrix(molecule, alpha_MOs, beta_MOs):
    alpha_density = HF.makeTemplateMatrix(len(alpha_MOs))
    beta_density = HF.makeTemplateMatrix(len(alpha_MOs))
    alpha_density = HF.make_density_matrix(alpha_density, alpha_MOs, molecule.NAlphaElectrons)
    beta_density = HF.make_density_matrix(beta_density, beta_MOs, molecule.NBetaElectrons)
    average_density = (alpha_density + beta_density) / 2.
    average_density = numpy.round(average_density, decimals = 5)
    average_density = numpy.ndarray.tolist(average_density)
    return average_density

def getNewDict(density, system,molecule):
    basis = system.BasisSets[-1]
    atom = molecule.Atoms[0].Label.upper()
    if not(basis in SADget):
        SADget[basis] = {}
    SADget[basis][atom] = density
    return SADget

def newLine(tab, File, nesting):
    File.write('\n')
    poss = nesting * tab
    File.write(' ' * poss)
    return poss 

def  writeToFile(outfile, new_dict):
    line_length = 100
    tab = 9    #tab size 
    filename = outfile + ".py"
    File = open(filename ,'w')
    text = str(new_dict)
    File.write("SADget = ")
    nesting = 1
    brace_nesting = 0
    poss = 0 
    for i in text:
        poss += 1 
        File.write(i)

        if i == '{': nesting += 1 
        elif i == '}': nesting -= 1
        elif i == '[': brace_nesting += 1 
        elif i == ']': brace_nesting -=1 

        #writing newlines for formating 
        if i == ',':        
            if nesting == 2:               #placing the start of each basis on a new line 
                poss = newLine(tab, File, 1)
            elif brace_nesting == 0:       # placing each new atom on a new line 
                poss = newLine(tab, File, 2)
            elif poss > line_length:        # wraping text 
                poss = newLine(tab, File, nesting)
        
    return 0

def Write(outfile, system, molecule, alpha_MOs, beta_MOs):
   density = getMatrix(molecule, alpha_MOs, beta_MOs)
   new_dict = getNewDict(density,system,molecule)
   writeToFile(outfile, new_dict)
   return 0 

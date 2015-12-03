# System 
Method = 'HF'
JobType = 'Energy'
BasisFit = True
BasisSets = ['STO3G']
#BasisSets = ['6311G']
#BasisSets = ['6-31G**']
UseMOM = True 
MOM_Type = "mutable"
SCFGuess = "core"
#Excitations = [[-1,1]] # HOMO -> LUMO (last of occ -> first of unocc), only for single electron excitations from reference
UseDIIS = False 
#Excitations = 'Single' # Approximates CIS (single electron transition energies)
#Excitations = 'Double' # Approximate CID matrix elements -> need to define Method for this
# Molecule
Charge = 1 
Multiplicity = 1
#Multiplicity = 3
#Coords = [['C', 6.0, 0.0, 0.0, 0.0],
#          ['H', 1.0, 0.0, 0.0, 1.0],
#          ['H', 1.0, 0.0, 1.0, 0.0],
#          ['H', 1.0, 1.0, 0.0, 0.0]]
#Coords = [['Li', 3.0, 0.0, 0.0, 0.0],['Li', 3.0, 0.0, 0.0, 1.4]]
#Coords = [['Li', 3.0, 0.0, 0.0, 0.0],['H', 1.0, 0.0, 0.0, 0.74]]
#Coords = [['H', 1.0, 0.0, 0.0, 0.0],['H', 1.0, 0.0, 0.0, 0.74]]
#Coords = [['He', 2.0, 0.0, 0.0, 0.0]]
Coords = [['He', 2.0, 0.0, 0.0, 0.0],['H', 1.0, 0.0, 0.0, 0.774]]
#Coords = [['Li', 3., 0.,0.,0.]]

[calc1]
BasisSets = ['STO-3G']
Multiplicity = 2
Charge = 0
Excitations = [[-1,1]]
direct = False
reference = "cuhf"
UseMOM = True
MOM_Type = "fixed"
SCFGuess = "Core"
MaxIterations = 50
MOFileWrite = "MOs.out"

# A second calculation
[calc2]
BasisSets = ['STO-3G', '321G']
Multiplicity = 1
Charge = 0
Excitations = [[-1,1]]
UseDISS = True

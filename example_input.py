[calc1]
BasisSets = ['STO-3G']
Multiplicity = 2
Charge = 0
Coords = [['H',1.0,  0.  ,  0.,   0.],
          ['H',1.0,  1.0 ,  0.,   0.],
          ['H',1.0,  0.4,  1.0,   0.]]
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
Coords = [['Li',3,0,0,0],['H',1,0,0,1.599427]]
Multiplicity = 1
Charge = 0
Excitations = [[-1,1]]
UseDISS = True
SCFGuess = "core"

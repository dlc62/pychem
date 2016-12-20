[Basis_Fitting_Test]

# This is an input file for generating the molecule structure requred to run
# tests of the basis fitting code

Charge = 0
Multiplicity = 1
Coords = [['Li', 3, 0,0,0], ['H',1, 0,0, 1.8]]

Excitations = "Single"

Method = "HF"
Basis_Sets = ["STO3G"]
SCF_Guess = "Core"
Reference = "UHF"

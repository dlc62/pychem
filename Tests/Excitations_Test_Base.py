[Excitations_test]

# This is an input file for generating the molecule structure requred to run
# tests for the excitations code, when tests run extra lines specifying the
# excitations are added to this file

Charge = 0
Multiplicity = 2
Coords = [['H' ,1.0 ,0.0  ,0.0   ,0.0 ],
          ['H' ,1.0 ,1.09 ,0.0   ,0.0 ],
          ['H', 1.0 ,.547 ,0.945 ,0.0]]
Method = "HF"
Basis_Sets = ["STO3G"]
SCF_Guess = "Core"
Reference = "UHF"

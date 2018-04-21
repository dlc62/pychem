# System libraries
import numpy
from numpy import dot

# Custom-written code
from Util import printf

#=================================================================#
#                                                                 #
#                        Main Function                            #
#                                                                 #
#=================================================================#

def do(settings, molecule):

  printf.delimited_text(settings.OutFile, " MP2 calculations for all electronic states ")

  for state_index, state in enumerate(molecule.States):

    # Load up MO coefficients and energies for current electronic state
    Ca = state.Alpha.MOs
    Cb = state.Beta.MOs
    Ea = state.Alpha.Energies
    Eb = state.Beta.Energies

    # Perform transformation of coulomb and exchange matrices from AO to MO basis

    # Allocate space to store transformed orbitals (in MO basis)

    Alpha_Coulomb = numpy.zeros((molecule.NOrbitals,) * 4)
    Beta_Coulomb = numpy.zeros((molecule.NOrbitals,) * 4)
    Alpha_Beta_Coulomb = numpy.zeros((molecule.NOrbitals,) * 4)
    Alpha_Exchange = numpy.zeros((molecule.NOrbitals,) * 4)
    Beta_Exchange = numpy.zeros((molecule.NOrbitals,) * 4)

    # Do brute-force (N^8) transformation, express each MO (a,b,c,d) as a linear combination of AOs (m,n,l,s)
    for a in range(0, molecule.NOrbitals):
      for b in range(0, molecule.NOrbitals):
        for c in range(0, molecule.NOrbitals):
          for d in range(0, molecule.NOrbitals):

            for m in range(0, molecule.NOrbitals):
              for n in range(0, molecule.NOrbitals):
                for l in range(0, molecule.NOrbitals):
                  for s in range(0, molecule.NOrbitals):
                    Alpha_Coulomb[a][b][c][d]      +=  Ca[m][a]*Ca[n][b]*Ca[l][c]*Ca[s][d] * molecule.CoulombIntegrals[m][n][l][s]
                    Beta_Coulomb[a][b][c][d]       +=  Cb[m][a]*Cb[n][b]*Cb[l][c]*Cb[s][d] * molecule.CoulombIntegrals[m][n][l][s]
                    Alpha_Beta_Coulomb[a][b][c][d] +=  Ca[m][a]*Ca[n][b]*Cb[l][c]*Cb[s][d] * molecule.CoulombIntegrals[m][n][l][s]
                    Alpha_Exchange[a][b][c][d]     +=  Ca[m][a]*Ca[s][d]*Ca[n][b]*Ca[l][c] * molecule.CoulombIntegrals[m][n][l][s]
                    Beta_Exchange[a][b][c][d]      +=  Cb[m][a]*Cb[s][d]*Cb[n][b]*Cb[l][c] * molecule.CoulombIntegrals[m][n][l][s]

    # Use the transformed integrals in the MP2 energy expression, i,j index occupied MOs and p,q are virtuals
    MP2_Eaa = 0.0
    MP2_Eab = 0.0
    MP2_Ebb = 0.0

    for i in range(0, molecule.NAlphaOrbitals):
      for j in range(0, i+1):
        for p in range(molecule.NAlphaOrbitals, molecule.NOrbitals):
          for q in range(molecule.NAlphaOrbitals, p+1):
            MP2_Eaa += (Alpha_Coulomb[i][p][j][q] - Alpha_Exchange[i][q][j][p])**2 / (Ea[i] + Ea[j] - Ea[p] - Ea[q])

    for i in range(0, molecule.NBetaOrbitals):
      for j in range(0, i+1):
        for p in range(molecule.NBetaOrbitals, molecule.NOrbitals):
          for q in range(molecule.NBetaOrbitals, p+1):
            MP2_Ebb += (Alpha_Coulomb[i][p][j][q] - Alpha_Exchange[i][q][j][p])**2 / (Eb[i] + Eb[j] - Eb[p] - Eb[q])

    for i in range(0, molecule.NAlphaOrbitals):
      for j in range(0, molecule.NBetaOrbitals):
        for p in range(molecule.NAlphaOrbitals, molecule.NOrbitals):
          for q in range(molecule.NBetaOrbitals, molecule.NOrbitals):
            MP2_Eab += (Alpha_Beta_Coulomb[i][p][j][q])**2 / (Ea[i] + Eb[j] - Ea[p] - Eb[q])

    MP2_Total_Energy = MP2_Eaa + MP2_Eab + MP2_Ebb

    printf.text_value(settings.OutFile," State: ", state_index, " Total MP2 energy: ", MP2_Total_Energy) 

  printf.delimited_text(settings.OutFile, " End of MP2 calculations ")

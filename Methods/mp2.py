# System libraries
import numpy

# Custom-written code
from Util import printf

#=================================================================#
#                                                                 #
#                        Main Function                            #
#                                                                 #
#=================================================================#

# Use states=None for calculations based on a set of HF states, pass in other states 
# if using e.g. as a correction to NOCI matrix elements

def do(settings, molecule, states=None):
  
  # For MP2 at the end of HF we do all the states and print the results 
  if states == None:
    printf.delimited_text(settings.OutFile, " MP2 calculations for all electronic states ")
    states = molecule.States

  mp2_energies = []

  for state_index, state in enumerate(states):

    # Load up MO coefficients and energies for current electronic state
    Ca = state.Alpha.MOs
    Cb = state.Beta.MOs
    Ea = state.Alpha.Energies
    Eb = state.Beta.Energies

    # Perform transformation of coulomb and exchange matrices from AO to MO basis

    # Allocate space to store half-transformed and fully-transformed orbitals (in MO basis)

    Alpha = numpy.zeros((molecule.NOrbitals,) * 4)
    Beta = numpy.zeros((molecule.NOrbitals,) * 4)
    AlphaBeta = numpy.zeros((molecule.NOrbitals,) * 4)

    # Do first half-transformation

    for m in range(0, molecule.NOrbitals):
      for l in range(0, molecule.NOrbitals):

        X = molecule.CoulombIntegrals[m,:,l,:]

        for b in range(0, molecule.NOrbitals):
          for d in range(0, molecule.NOrbitals):
            Alpha[m,b,l,d]     = (Ca[:,b].T).dot(X.dot(Ca[:,d]))  
            Beta[m,b,l,d]      = (Cb[:,b].T).dot(X.dot(Cb[:,d]))  
            AlphaBeta[m,b,l,d] = (Ca[:,b].T).dot(X.dot(Cb[:,d]))  
 
    # Complete transformation replacing m,b,l,d elements with a,b,c,d

    for b in range(0, molecule.NOrbitals):
      for d in range(0, molecule.NOrbitals):
        
        # TODO why are these being copied?
        Yaa = Alpha[:,b,:,d].copy()
        Yab = AlphaBeta[:,b,:,d].copy()
        Ybb = Beta[:,b,:,d].copy()

        for a in range(0, molecule.NOrbitals):
          for c in range(0, molecule.NOrbitals):

            Alpha[a,b,c,d]     = (Ca[:,a].T).dot(Yaa.dot(Ca[:,c])) 
            Beta[a,b,c,d]      = (Cb[:,a].T).dot(Ybb.dot(Cb[:,c])) 
            AlphaBeta[a,b,c,d] = (Ca[:,a].T).dot(Yab.dot(Cb[:,c])) 

    # Use the transformed integrals in the MP2 energy expression, i,j index occupied MOs and p,q are virtuals
    MP2_Eaa = 0.0
    MP2_Eab = 0.0
    MP2_Ebb = 0.0


    if not "P2-SOS" in settings.Method:
      for i in range(0, molecule.NAlphaElectrons):
        for j in range(0, i+1):
          for p in range(molecule.NAlphaElectrons, molecule.NOrbitals):
            for q in range(molecule.NAlphaElectrons, p+1):
              MP2_Eaa += (Alpha[i,p,j,q] - Alpha[i,q,j,p])**2 / (Ea[i] + Ea[j] - Ea[p] - Ea[q])

      for i in range(0, molecule.NBetaElectrons):
        for j in range(0, i+1):
          for p in range(molecule.NBetaElectrons, molecule.NOrbitals):
            for q in range(molecule.NBetaElectrons, p+1):
              MP2_Ebb += (Beta[i,p,j,q] - Beta[i,q,j,p])**2 / (Eb[i] + Eb[j] - Eb[p] - Eb[q])

    for i in range(0, molecule.NAlphaElectrons):
      for j in range(0, molecule.NBetaElectrons):
        for p in range(molecule.NAlphaElectrons, molecule.NOrbitals):
          for q in range(molecule.NBetaElectrons, molecule.NOrbitals):
              MP2_Eab += (AlphaBeta[i,p,j,q])**2 / (Ea[i] + Eb[j] - Ea[p] - Eb[q])

    # Spin component scaling
    if "P2-SCS" in settings.Method:
        print("Doing SCS ")
        MP2_Eaa *= 1/3 
        MP2_Ebb *= 1/3 
        MP2_Eab *= 6/5 

    elif "P2-SOS" in settings.Method:
      print("Doing SOS")
      MP2_Eaa *= 0 
      MP2_Ebb *= 0 
      MP2_Eab *= 1.3

    MP2_Total_Energy = MP2_Eaa + MP2_Eab + MP2_Ebb

    #if states == None: 
    #printf.text_value(settings.OutFile," State: ", state_index, " Total MP2 energy: ", MP2_Total_Energy) 
    printf.text_value(settings.OutFile," State: ", state_index, " Total MP2 energy: ", state.TotalEnergy + MP2_Total_Energy) 

    mp2_energies.append(MP2_Total_Energy)

  #if states == None:
  printf.delimited_text(settings.OutFile, " End of MP2 calculations ")

  return MP2_Total_Energy

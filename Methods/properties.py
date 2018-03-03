from __future__ import print_function
import numpy
from hartree_fock import evaluate_2e_ints
from Util import printf

def calculate(settings, molecule):

   if settings.PropertyType == 'SCATTERING':

      printf.delimited_text(settings.OutFile," Property calculation - electron scattering intensities ")

      # Set up structures to hold outputs, compute intermediates
      scattering_patterns = [[] for i in range(0,molecule.NStates)]
      two_pdms = [make_two_particle_density_matrix(molecule.NOrbitals, state) for state in molecule.States]
#      print('overlap integrals')
      coulomb,exchange = check_normalization(molecule.NOrbitals, two_pdms[0], molecule.Overlap)
#      print('coulomb and exchange normalization',coulomb, exchange)

      # For each grid point, generate scattering integrals and contract with 2-PDM for each electronic state
#      print('scattering integrals')
      for grid_point in settings.PropertyGrid:
         evaluate_2e_ints(molecule, 1, grid_point)
         for index,state in enumerate(molecule.States):
            value = contract_two(molecule.NOrbitals, two_pdms[index], molecule.CoulombIntegrals, molecule.ExchangeIntegrals)
#            print('coulomb + exchange scattering', value)
            scattering_patterns[index].append(molecule.NElectrons+value)

      # Print out results
      outstring = "Grid value -> Scattering patterns for each electronic state \n\n"
      for i,grid_point in enumerate(settings.PropertyGrid):
         outstring += "%10.6f" % grid_point
         for index in range(0,molecule.NStates):
            outstring += "%16.12f" % scattering_patterns[index][i] 
         outstring += "\n"
      printf.print_to_file(settings.OutFile, outstring+'\n') 

   printf.delimited_text(settings.OutFile," End of property calculation ")

#-------------------------------------------------------------

def make_two_particle_density_matrix(n_orbitals, this):
   
    two_pdm = numpy.zeros((n_orbitals,) * 4)
 
    for a in range(0,n_orbitals):
      for b in range(0,n_orbitals):
        for c in range(0,n_orbitals):
          for d in range(0,n_orbitals):

            two_pdm[a,b,c,d] = (this.Alpha.Density[a,b]*this.Alpha.Density[c,d]
                                +this.Beta.Density[a,b] *this.Beta.Density[c,d]) 

    return two_pdm

#-------------------------------------------------------------

def contract_two(n_orbitals, two_pdm, coulomb, exchange):

    value = 0.0

    for a in range(0,n_orbitals):
      for b in range(0,n_orbitals):
        for c in range(0,n_orbitals):
          for d in range(0,n_orbitals):

            value += two_pdm[a,b,c,d]*(2*coulomb[a,b,c,d]-exchange[a,d,c,b])
#            print(a,b,c,d, coulomb[a,b,c,d], exchange[a,d,c,b])
#            value += two_pdm[a,b,c,d]*(coulomb[a,b,c,d])

    return value

#-------------------------------------------------------------

def check_normalization(n_orbitals, two_pdm, overlap):

    coulomb = 0.0; exchange = 0.0

    for a in range(0,n_orbitals):
      for b in range(0,n_orbitals):
        for c in range(0,n_orbitals):
          for d in range(0,n_orbitals):

            coulomb += two_pdm[a,b,c,d]*overlap[a,b]*overlap[c,d]
            exchange -= 0.5*two_pdm[a,b,c,d]*overlap[a,d]*overlap[c,b]
#            print(a,b,c,d,overlap[a,b]*overlap[c,d],overlap[a,d]*overlap[c,b])

    return coulomb,exchange

#-------------------------------------------------------------

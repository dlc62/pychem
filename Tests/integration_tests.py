
# Run tests from the main pychem dirrectory 

import unittest 
import ConfigParser
import sys 
import os 
import numpy as np

# Add the parent dir to the path so we can import pychem 
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pychem

class IntegrationTest():

    @classmethod
    def setUpClass(self):
        self.molecule = pychem.main(self.infile_name)


    @classmethod
    def tearDownClass(self):
        parser = ConfigParser.SafeConfigParser()
        parser.read(self.infile_name)
        for section in parser.sections():
            outfile = section + ".out"
            if os.path.exists(outfile):
                os.remove(outfile)


    # test that the calculation ran without crashing
    def test_run(self):
        self.molecule


    def HF_energies_test(self, energies):
        for i, energy in enumerate(energies):
            np.testing.assert_almost_equal(energy, self.molecule.States[i].TotalEnergy, decimal=5)


    def NOCI_energies_test(self, energies):
        for i, energy in enumerate(energies):
           np.testing.assert_almost_equal(energy, self.molecule.NOCIEnergies[i], decimal=5)


    # takes an arrayy and compares that to the upper right corner 
    # of the molecule alpha orbitals
    def alpha_orbitals_test(self, state_index, sub_array):
        x,y = sub_array.shape
        sub_orbitals = self.molecule.States[state_index].Alpha.MOs[:x, :y]
        np.testing.assert_almost_equal(sub_array, sub_orbitals, decimal=5)


# Testing LiH at 2.2A in 6-31G
class LiH_SFS_NOCI_Test(IntegrationTest, unittest.TestCase):
    infile_name = "Tests/LiH_SFS_NOCI.test.inp"


    def test_HF_energies(self):
        self.HF_energies_test([-7.957898,-7.915383])


    def test_NOCI_energies(self):
        self.NOCI_energies_test([-7.96647127, -7.91538363, -7.86526525])


    # This runs after NOCI has rearanged the orbitals
    def test_alpha_orbitals(self):
        sub_array = np.array([[0.99901, 0.06856],
                              [0.01636,-0.09058]])
        self.alpha_orbitals_test(0, sub_array)


# Testing H2 at 1.0A in 6-311G
class H2_Test(IntegrationTest, unittest.TestCase):
    infile_name = "Tests/H2_HF.test.inp"


    def test_HF_energies(self):
        self.HF_energies_test([-1.096864])




if __name__ == '__main__':
    print("\nStarting integration tests, this could take a while\n")
    unittest.main()



#! /usr/bin/env python3

# Regression tests for the basis fitting code

# Run the tests with python3 -m unittest basis_fitting.tests

# Note the example MOs don't have involve addinf polarizing functions

import unittest
import numpy as np
import configparser

from Methods import basis_fit
from Util import inputs_structures, util


# Converged STO3G alpha orbitals for the ground state of LiH at 1.8A

LiH_MOs = np.array([[ -9.91372e-01,   1.56976e-01,   2.12299e-01,  -3.64763e-17,  -1.15146e-16,  -1.09447e-01],
                    [ -3.33462e-02,  -4.71939e-01,  -7.98087e-01,   2.19600e-16,   5.05756e-16,   6.27657e-01],
                    [ -6.61948e-18,  -5.83991e-17,  -2.42615e-16,   7.22098e-01,  -6.91791e-01,   1.00763e-16],
                    [ -5.71842e-17,   2.62139e-16,  -4.37608e-16,  -6.91791e-01,  -7.22098e-01,   2.57926e-16],
                    [  5.20412e-03,  -3.38928e-01,   6.07188e-01,  -1.06816e-16,  -2.37020e-16,   9.23754e-01],
                    [ -9.56834e-04,  -5.63445e-01,   1.58278e-01,  -5.72359e-17,  -1.48825e-16,  -1.08581e+00]])

# The above orbitals fit to 321G
LiH_MOs_321G = np.array([[ -9.89609e-01,   1.29589e-01,   1.66182e-01,  -2.38676e-17,  -8.59919e-17,  -7.33893e-02,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00],
                         [ -4.65651e-02,  -3.27376e-01,  -5.54862e-01,   1.53188e-16,   3.52075e-16,   4.37717e-01,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00],
                         [ -1.27198e-03,  -1.59907e-01,  -2.69885e-01,   7.40415e-17,   1.70835e-16,   2.11676e-01,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00],
                         [ -6.10099e-18,  -5.38248e-17,  -2.23611e-16,   6.65537e-01,  -6.37604e-01,   9.28709e-17,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00],
                         [ -5.27051e-17,   2.41607e-16,  -4.03331e-16,  -6.37604e-01,  -6.65537e-01,   2.37723e-16,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00],
                         [  4.79649e-03,  -3.12380e-01,   5.59628e-01,  -9.84497e-17,  -2.18455e-16,   8.51398e-01,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00],
                         [ -7.77004e-19,  -6.85497e-18,  -2.84785e-17,   8.47609e-02,  -8.12035e-02,   1.18278e-17,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00],
                         [ -6.71237e-18,   3.07703e-17,  -5.13671e-17,  -8.12035e-02,  -8.47609e-02,   3.02757e-17,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00],
                         [  6.10867e-04,  -3.97839e-02,   7.12727e-02,  -1.25383e-17,  -2.78218e-17,   1.08432e-01,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00],
                         [ -5.46895e-04,  -3.22047e-01,   9.04664e-02,  -3.27141e-17,  -8.50634e-17,  -6.20615e-01,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00],
                         [ -5.07394e-04,  -2.98786e-01,   8.39323e-02,  -3.03513e-17,  -7.89195e-17,  -5.75790e-01,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00]])

class BasisFitTests(unittest.TestCase):

    # Test the idempotence of fitting a set onto itself
    def setUp(self):
        """ Create a molecule object """
        parser = configparser.ConfigParser()
        parser.read("Tests/Basis_Fitting_Test_Example.py")
        self.molecule, _ = inputs_structures.process_input(parser.sections()[0], parser)

    def test_idem(self):
        """ Test the idempotence of the method """
        equal = np.allclose(basis_fit.do(self.molecule, LiH_MOs, "STO3G"), LiH_MOs)
        self.assertTrue(equal)

    def test_overall(self):
        """ Test the method as a whole works """
        equal = np.allclose(basis_fit.do(self.molecule, LiH_MOs, "321G"), LiH_MOs_321G)
        self.assertTrue(equal)

    def test_atom(self):
        """ test the basis_fit atom function """
        atom_coeffs = basis_fit.Basis_Fit_Atom(self.molecule.Atoms[0], list(LiH_MOs[:,0]), 0, '321G')
        equal = np.allclose(atom_coeffs, list(LiH_MOs_321G[:9,0]))
        self.assertTrue(equal)

    def test_ang(self):
        """ test the fit angular momentum function """
        atom = self.molecule.Atoms[0]
        old_ang_set = atom.Basis[:2]
        MOs = list(LiH_MOs[:,0])
        new_ang_set = [[[36.8382, 0.0696686], [5.48172, 0.381346], [1.11327, 0.681702]], [[0.540205, -0.263127], [0.102255, 1.14339]], [[0.028565, 1.0]]]
        ang_coeffs = basis_fit.Basis_Fit_Ang(atom, old_ang_set, MOs, 0, new_ang_set, 0)
        equal = np.allclose(ang_coeffs, LiH_MOs_321G[:3,0])
        self.assertTrue(equal)

    def _test_back(self):
        """Testing fitting a larger set onto a smaller one"""
        print(LiH_MOs_321G)
        print('')
        result = basis_fit.do(self.molecule, LiH_MOs_321G, "STO3G")
        print(result)
        print('')
        result2 = basis_fit.do(self.molecule, result, "321G")
        print(result2)
        print('')

class SubFunctionTests(unittest.TestCase):

    def setUp(self):
        """ Create a molecule object """
        parser = configparser.ConfigParser()
        parser.read("Tests/Basis_Fitting_Test_Example.py")
        self.molecule, _ = inputs_structures.process_input(parser.sections()[0], parser)
        import pdb; pdb.set_trace()

    def test_get_indices(self):
        expected_result = [[0,1,5], [2,3,4,6,7,8]]
        result = basis_fit.get_ang_indices(self.molecule.Atoms[0], "321G")
        self.assertEqual(result, expected_result)

parser = configparser.ConfigParser()
parser.read("Tests/Basis_Fitting_Test_Example.py")
molecule, _ = inputs_structures.process_input(parser.sections()[0], parser)
result = basis_fit.do(molecule, LiH_MOs, "321G")

print("Original STO3G MOs")
util.visualize_MOs(LiH_MOs, "STO3G", molecule)
print("Basis Fit 321G MOs")
util.visualize_MOs(LiH_MOs_321G, "321G", molecule)
print("Basis Fit 321G MOs")
util.visualize_MOs(result, "321G", molecule)

#suite = unittest.TestLoader().loadTestsFromTestCase(SubFunctionTests)
#unittest.TextTestRunner(verbosity=1).run(suite)

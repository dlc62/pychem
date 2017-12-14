#! /usr/bin/env python3

# Regression tests for the basis fitting code

# Run the tests with python3 -m unittest Tests.basis_fitting_tests

# Note the example MOs don't have involve adding polarizing functions

import unittest
import numpy as np
import configparser

import numpy

from Methods import basis_fit
from Util import inputs_structures, util
from Data import basis


# Converged STO3G alpha orbitals for the ground state of LiH at 1.8A

LiH_MOs = np.array([[ -9.91372e-01,   1.56976e-01,   2.12299e-01,  -3.64763e-17,  -1.15146e-16,  -1.09447e-01],
                    [ -3.33462e-02,  -4.71939e-01,  -7.98087e-01,   2.19600e-16,   5.05756e-16,   6.27657e-01],
                    [ -6.61948e-18,  -5.83991e-17,  -2.42615e-16,   7.22098e-01,  -6.91791e-01,   1.00763e-16],
                    [ -5.71842e-17,   2.62139e-16,  -4.37608e-16,  -6.91791e-01,  -7.22098e-01,   2.57926e-16],
                    [  5.20412e-03,  -3.38928e-01,   6.07188e-01,  -1.06816e-16,  -2.37020e-16,   9.23754e-01],
                    [ -9.56834e-04,  -5.63445e-01,   1.58278e-01,  -5.72359e-17,  -1.48825e-16,  -1.08581e+00]])

# The above orbitals fit to 321G
LiH_MOs_321G = np.array([[ -1.00228e+00,   1.57718e-01,   2.12973e-01,  -3.64219e-17,  -1.15361e-16,  -1.09348e-01,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00],
                         [ -3.64789e-02,  -3.74580e-01,  -6.33976e-01,   1.74663e-16,   4.01951e-16,   4.99166e-01,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00],
                         [ -6.18446e-18,  -5.45612e-17,  -2.26671e-16,   6.74643e-01,  -6.46328e-01,   9.41411e-17,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00],
                         [ -5.34262e-17,   2.44912e-16,  -4.08849e-16,  -6.46328e-01,  -6.74643e-01,   2.40976e-16,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00],
                         [  4.86212e-03,  -3.16654e-01,   5.67285e-01,  -9.97963e-17,  -2.21444e-16,   8.63047e-01,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00],
                         [ -7.11693e-03,  -1.25447e-01,  -2.12049e-01,   5.83087e-17,   1.34344e-16,   1.66666e-01,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00],
                         [ -7.69573e-19,  -6.78941e-18,  -2.82061e-17,   8.39502e-02,  -8.04268e-02,   1.17146e-17,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00],
                         [ -6.64816e-18,   3.04760e-17,  -5.08758e-17,  -8.04268e-02,  -8.39502e-02,   2.99862e-17,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00],
                         [  6.05025e-04,  -3.94033e-02,   7.05909e-02,  -1.24183e-17,  -2.75557e-17,   1.07394e-01,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00],
                         [ -5.62609e-04,  -3.31300e-01,   9.30659e-02,  -3.36541e-17,  -8.75076e-17,  -6.38446e-01,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00],
                         [ -5.00092e-04,  -2.94486e-01,   8.27245e-02,  -2.99145e-17,  -7.77839e-17,  -5.67502e-01,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00,   0.00000e+00]])

# making assert_allclose behave more like numpy's allclose function
assert_allclose = (lambda x,y: numpy.testing.assert_allclose(x,y, rtol=1e-05, atol=1e-08))

class BasisFitTests(unittest.TestCase):

    def setUp(self):
        """ Create a molecule object """
        parser = configparser.ConfigParser()
        parser.read("Tests/Basis_Fitting_Test_Example.py")
        self.molecule, _ = inputs_structures.process_input(parser.sections()[0], parser)

    def test_idem(self):
        """ Test that fitting orbitals onto the same basis returns the original orbitals"""
        assert_allclose(basis_fit.do(self.molecule, LiH_MOs, "STO3G"), LiH_MOs)

    def test_overall(self):
        """ Test the method as a whole works """
        equal = np.allclose(basis_fit.do(self.molecule, LiH_MOs, "321G"), LiH_MOs_321G)
        self.assertTrue(equal)

    def test_atom(self):
        """ test the basis_fit atom function """
        atom_coeffs = basis_fit.Basis_Fit_Atom(self.molecule.Atoms[1], LiH_MOs[:,0], 5, '321G', 'STO3G')
        assert_allclose(atom_coeffs, list(LiH_MOs_321G[9:,0]))

    def test_ang(self):
        """ test the fit angular momentum function """
        atom = self.molecule.Atoms[0]
        old_ang_set = atom.Basis[:2]
        MOs = LiH_MOs[:,0]
        new_ang_set = [[[36.8382, 0.0696686], [5.48172, 0.381346], [1.11327, 0.681702]], [[0.540205, -0.263127], [0.102255, 1.14339]], [[0.028565, 1.0]]]
        ang_coeffs = basis_fit.Basis_Fit_Ang(atom, old_ang_set, MOs[[0,1]], new_ang_set, 0)
        assert_allclose(ang_coeffs, LiH_MOs_321G[[0,1,5],0])


    def test_back(self):
        """Testing fitting a larger set onto a smaller one. Currently this just checks there
           are no exceptions thrown"""
        result = basis_fit.do(self.molecule, LiH_MOs_321G, "STO3G")

class SubFunctionTests(unittest.TestCase):

    def setUp(self):
        """ Create a molecule object """
        parser = configparser.ConfigParser()
        parser.read("Tests/Basis_Fitting_Test_Example.py")
        self.molecule, _ = inputs_structures.process_input(parser.sections()[0], parser)

    def test_get_indices(self):
        expected_result = [[0,1,5], [2,3,4,6,7,8]]
        cgtos = basis.get["321G"]["LI"]
        result = basis_fit.get_ang_indices(self.molecule.Atoms[0], cgtos)
        self.assertEqual(result, expected_result)


parser = configparser.ConfigParser()
parser.read("Tests/Basis_Fitting_Test_Example.py")
molecule, _ = inputs_structures.process_input(parser.sections()[0], parser)
result = basis_fit.do(molecule, LiH_MOs, "321G")

#print("Original STO3G MOs")
#util.visualize_MOs(LiH_MOs, "STO3G", molecule)
#print("Basis Fit 321G MOs")
#util.visualize_MOs(LiH_MOs_321G, "321G", molecule)
#print("Basis Fit 321G MOs")
#util.visualize_MOs(result, "321G", molecule)


#suite = unittest.TestLoader().loadTestsFromTestCase(SubFunctionTests)
#unittest.TextTestRunner(verbosity=1).run(suite)

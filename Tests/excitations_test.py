import unittest
import configparser

from os import remove

from Util import inputs_structures

input_filepath = "Tests/excitations_test_input.py"
base_file = "Tests/Excitations_Test_Base.py"

def make_input_file(input_excitations):
    with open(base_file, "r") as f:
        lines = f.readlines()
    for feild, value in input_excitations.items():
        lines.append(feild + "=" + str(value) + "\n")
    with open(input_filepath, "w+") as out:
        for line in lines:
            out.write(line)

def setup_test(input_excitations):
        make_input_file(input_excitations)
        parser = configparser.ConfigParser()
        parser.read(input_filepath)
        molecule, _ = inputs_structures.process_input(parser.sections()[0], parser)
        remove(input_filepath)
        return molecule

class ExcitationsTest(unittest.TestCase):

    def make_test(input_excitations, expected_output):
        def new_test(self):
            # Create and read the input
            molecule = setup_test(input_excitations)
            alpha_states = [state.AlphaOccupancy for state in molecule.States]
            beta_states = [state.BetaOccupancy for state in molecule.States]

            # Gather up the generated excitations ignoring the ground state
            # Not worrying about order
            self.assertCountEqual(expected_output["alpha"], alpha_states[1:])
            self.assertCountEqual(expected_output["beta"], beta_states[1:])

        return new_test

    # Manually Specifying single excitations
    test_manual_single  = make_test({"alpha_excitations": [[0,2], [1,2]],
                                     "beta_excitations":  [[]]},
                                    {"alpha": [[0,1,1], [1,0,1]],
                                     "beta":  [[1,0,0], [1,0,0]]})

    # Test manual specification of double excitations
    test_manual_double = make_test({"alpha_excitations": [[0,2]],
                                    "beta_excitations": [[0,1]]},
                                   {"alpha": [[0,1,1]],
                                    "beta": [[0,1,0]]})

    # Test mixed spin single excitations
    test_mixed = make_test({"alpha_excitations": [[0,2]],
                            "beta_excitations": [[], [0,1]]}, # using empty list as placeholder
                           {"alpha": [[0,1,1], [1,1,0]],
                            "beta": [[1,0,0], [0,1,0]]})

    # No Excitations
    test_none = make_test({"foo": "'bar'",
                           "waed": "'dags'"},
                          {"alpha": [],
                           "beta": []})


    ###### Keywords ######

    # Single
    test_single = make_test({"Excitations": "'SINGLE'"},
                            {"alpha": [[0,1,1], [1,0,1], [1,1,0], [1,1,0]],
                             "beta":  [[1,0,0], [1,0,0], [0,0,1], [0,1,0]]})

    # HOMO-LUMO Excitations
    test_HOMO_LUMO = make_test({"Excitations": "'HOMO-LUMO'"},
                               {"alpha": [[1,0,1]],
                                "beta": [[1,0,0]]})

    # Double paired
    test_double_paired = make_test({"Excitations": "'DOUBLE-PAIRED'"},
                                   {"alpha": [[0,1,1]],
                                    "beta": [[0,0,1]]})

    ## All Double
    test_double = make_test({"Excitations": "'DOUBLE'"},
                            {"alpha": [[1,0,1], [1,0,1], [0,1,1], [0,1,1]],
                             "beta": [[0,1,0], [0,1,0], [0,0,1], [0,0,1]]})


# Test situations where input is rejected
#class ExcitationsTestErrors(unittest.TestCase):

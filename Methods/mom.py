# System libraries
import numpy

########################## MOM and Excited State Functions  ###########################

def do(molecule, this, state_index, ref_MOs):

    """ Reorders MO array and MO energies, placing columns in
        descending order of their overlap with the reference state"""

    # Load up reference orbitals
    alpha_ref_MOs = ref_MOs[0]
    beta_ref_MOs = ref_MOs[1]

    # Testing overlap againts other states currently isn't working well
    alpha_state_overlaps = numpy.zeros(molecule.NOrbitals)
    beta_state_overlaps = numpy.zeros(molecule.NOrbitals)
    NStates = 0    # keeps track of the number of states included

    # Calculate the overlap of the new MOs with all the states
    #for i, state in enumerate(molecule.States):                         # Using unoptimized states
    for i, state in enumerate(molecule.States[:state_index+1]):          # Only using previously optimized states

    # Overlap with the reference orbitals
        if i == state_index:
            alpha_p_vector = make_p_vector(this.Alpha.MOs, alpha_ref_MOs, molecule.NAlphaElectrons, molecule.Overlap)
            beta_p_vector = make_p_vector(this.Beta.MOs, beta_ref_MOs, molecule.NBetaElectrons, molecule.Overlap)

    # Calculate the overlap with just the orbtials that differ between each state
    # Should add to code to try and stop comparison with collapsed states
        #else:
        #    alpha_state_overlaps += compare_states(this.Alpha, state.Alpha, molecule.Overlap)
        #    beta_state_overlaps += compare_states(this.Beta, state.Beta, molecule.Overlap)
        #    NStates += 1

    # Take average of the state overlaps and subtract from the overlap
    # with the reference orbitals

    #if NStates is not 0:
    #    alpha_state_overlaps /= NStates    # Use 'NStates' for now to keep track of how the divisor chanages
    #    beta_state_overlaps /= NStates     # depending on whether the unoptimized states are used or not

    #    alpha_p_vector += alpha_state_overlaps
    #    beta_p_vector += beta_state_overlaps

    # Sort MOs according to p vector ordering
    this.Alpha.MOs, this.Alpha.Energies = Sort_MOs(this.Alpha.MOs, this.Alpha.Energies, alpha_p_vector)
    this.Beta.MOs, this.Beta.Energies = Sort_MOs(this.Beta.MOs, this.Beta.Energies, beta_p_vector)

def compare_states(new_state, old_state, overlap_matrix):
    """ Compare each of the MOs to the MOs that differ between the other state and
        the new state
        The two state agruments are actually the Matrix objects for the current spin"""

    # Get the differeces between the two states making use of Python's notion of truthiness
    # Note these are reversed since the orbitals have been swapped in the reference set
    states = range(len(overlap_matrix))
    excited_from = [i for i in states if new_state.Occupancy[i] and not old_state.Occupancy[i]]
    excited_to = [i for i in states if old_state.Occupancy[i] and not new_state.Occupancy[i]]

    # calculate the overlap matrix
    MO_overlaps = old_state.MOs.T.dot(overlap_matrix).dot(new_state.MOs)

    # extract the relevant blocks`
    to_overlaps = MO_overlaps[excited_to]
    from_overlaps = MO_overlaps[excited_from]

    # reduce the blocks
    excited_to_overlap = numpy.add.reduce(to_overlaps)
    excited_from_overlap = numpy.add.reduce(from_overlaps)

    # Subtract the magnitiude of the excited_from from that of excited_to
    return (numpy.abs(excited_to_overlap) - numpy.abs(excited_from_overlap))


def make_p_vector(new_MOs, other_MOs, NElectrons, overlap_matrix):
    """ Calculates the overlap vector between two sets of MOs """
    # MO_overlap = <ref|new>
    C_dagger = other_MOs[:,:NElectrons].T   
    MO_overlap = C_dagger.dot(overlap_matrix).dot(new_MOs)
    p_vector = numpy.add.reduce(MO_overlap)
    return numpy.abs(p_vector)

def Sort_MOs(MOs, energies, p):
    """Sorts MOs and energies in decending order
    based on a vector p (the overlap vector)"""
    indexes = p.argsort()[::-1]
    new_MOs = MOs[:,indexes]
    new_energies = energies[indexes]
    return new_MOs, new_energies

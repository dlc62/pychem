# System libraries
import numpy
from numpy import dot

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
    #NStates = 0    # keeps track of the number of states included

    # Calculate the overlap of the new MOs with all the states
    #for i, state in enumerate(molecule.States):                         # Using unoptimized states
    for i, state in enumerate(molecule.States[:state_index+1]):          # Only using previously optimized states

    # Overlap with the reference orbitals
        if i == state_index:
            alpha_p_vector = make_p_vector(this.Alpha.MOs, alpha_ref_MOs, molecule.NAlphaElectrons, molecule.Overlap)
            beta_p_vector = make_p_vector(this.Beta.MOs, beta_ref_MOs, molecule.NBetaElectrons, molecule.Overlap)

    # Calculate the overlap with just the orbtials that differ between each state
        else:
            alpha_state_overlaps += compare_states(this.Alpha, state.Alpha, molecule.Overlap)
            beta_state_overlaps += compare_states(this.Beta, state.Beta, molecule.Overlap)
            #NStates += 1

    # Take average of the state overlaps

        #alpha_state_overlaps /= NStates    # Use 'NStates' for now to keep track of how the divisor chanages
        #beta_state_overlaps /= NStates     # depending on whether the uoptimized states are used or not

        # subtract this from the overlap with the reference orbitals
    alpha_p_vector += alpha_state_overlaps
    beta_p_vector += beta_state_overlaps

    # Sort MOs according to p vector ordering
    this.Alpha.MOs, this.Alpha.Energies = Sort_MOs(this.Alpha.MOs, this.Alpha.Energies, alpha_p_vector)
    this.Beta.MOs, this.Beta.Energies = Sort_MOs(this.Beta.MOs, this.Beta.Energies, beta_p_vector)

def compare_states(new_state, old_state, overlap_matrix):

    """ Compare each of the MOs to the MOs that differ between the other state and
        the new state """

    # Get the differeces between the two states
    # There must be a nicer way to do this
    excited_to = []
    excited_from = []
    for i in range(len(overlap_matrix)):
        if new_state.Occupancy[i] and not old_state.Occupancy[i]:
            excited_to.append(i)
        elif not new_state.Occupancy[i] and old_state.Occupancy[i]:
            excited_from.append(i)

    # Find which MOs overlap with the orbitals excited from
    excited_from_overlap = numpy.zeros(len(new_state.MOs))
    for i in excited_from:
        pre_vector = old_state.MOs[:,i].T.dot(overlap_matrix)
        for j in range(len(new_state.MOs)):
            excited_from_overlap[j] += pre_vector.dot(new_state.MOs[:,j])

    # Find which MOs overlap with the states excited to
    excited_to_overlap = numpy.zeros(len(new_state.MOs))
    for i in excited_from:
        pre_vector = old_state.MOs[:,i].T.dot(overlap_matrix)
        for j in range(len(new_state.MOs)):
            excited_to_overlap[j] += pre_vector.dot(new_state.MOs[:,j])

    # Subtract the magnitiude of the excited_from from that of excited_to
    return numpy.abs(excited_to_overlap) - numpy.abs(excited_from_overlap)


def make_p_vector(new_MOs, other_MOs, NElectrons, overlap_matrix):
    """ Calculates the overlap vector between two sets of MOs """
    """ Note the old method of truncating the reference orbitals gave the correct
        size MO overlap matrix """

    # MO_overlap = <ref|new>
    C_dagger = other_MOs[:,:NElectrons].T
    MO_overlap= C_dagger.dot(overlap_matrix).dot(new_MOs)

    # Get the overlap with the reference orbitals
    # Overlap with the occupied space
    p_vector = numpy.zeros(len(new_MOs))
    for i in range(len(new_MOs)):
        p_vector[i] = sum(MO_overlap[:,i])

    return numpy.abs(p_vector)


def Sort_MOs(MOs, energies, p):
    """Sorts MOs and energies in decending order
    based on a vector p (the overlap vector)"""
    temp = [[p[i],MOs[:,i],energies[i]] for i in range(len(p))]
    temp = sorted(temp, key = lambda pair: pair[0], reverse = True)     #sorts the elements on the basis of the p values (temp[0])
    new_MOs = numpy.array([line[1] for line in temp])
    new_energies = [line[2] for line in temp]
    return numpy.transpose(new_MOs), new_energies

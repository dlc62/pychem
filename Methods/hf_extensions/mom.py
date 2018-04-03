# System libraries
import numpy

########################## MOM and Excited State Functions  ###########################

def do(molecule, this, state_index, ref_MOs):

    """ Reorders MO array and MO energies, placing columns in 
        descending order of their overlap with the reference state"""

    # Load up reference orbitals 
    alpha_ref_MOs = ref_MOs[0]
    beta_ref_MOs = ref_MOs[1]

    # Make alpha and beta p-vectors
    alpha_p_vector = make_p_vector(this.Alpha.MOs, alpha_ref_MOs, molecule.NElectrons, molecule.Overlap)
    beta_p_vector = make_p_vector(this.Beta.MOs, beta_ref_MOs, molecule.NElectrons, molecule.Overlap)

    # Store MOs according to p-vector ordering
    this.Alpha.MOs, this.Alpha.Energies = Sort_MOs(this.Alpha.MOs, this.Alpha.Energies, alpha_p_vector) 
    this.Beta.MOs, this.Beta.Energies = Sort_MOs(this.Beta.MOs, this.Beta.Energies, beta_p_vector) 


def make_p_vector(new_MOs, other_MOs, NElectrons, overlap_matrix):
    C_dagger = numpy.transpose(other_MOs[:,range(NElectrons)])
    MO_overlap = C_dagger.dot(overlap_matrix.dot(new_MOs))

    # Get the overlap with the reference orbitals
    P_vector = numpy.zeros(len(new_MOs))
    for i in range(len(new_MOs)):
        P_vector[i] = sum(MO_overlap[:,i])

    return numpy.abs(P_vector)


def Sort_MOs(MOs, energies, p):
    """Sorts MOs and energies in decending order
    based on a vector p (the overlap vector)"""
    temp = [[p[i],MOs[:,i],energies[i]] for i in range(len(p))]
    temp = sorted(temp, key = lambda pair: pair[0], reverse = True)     #sorts the elements on the basis of the p values (temp[0])
    new_MOs = numpy.array([line[1] for line in temp])
    new_energies = [line[2] for line in temp]
    return numpy.transpose(new_MOs), new_energies


# System libraries
import numpy as np

def do(molecule, this, ref_MOs):

    """ Reorders MO array and MO energies, placing columns in 
        descending order of their overlap with the reference state"""

    # Load up reference orbitals 
    alpha_ref_MOs = ref_MOs[0][:,:this.NAlpha]
    beta_ref_MOs = ref_MOs[1][:,:this.NBeta]

    # Make alpha and beta p-vectors
    alpha_p_vector = make_p_vector(this.Alpha.MOs, alpha_ref_MOs, molecule.Overlap)
    beta_p_vector = make_p_vector(this.Beta.MOs, beta_ref_MOs, molecule.Overlap)

    # Store MOs according to p-vector ordering
    this.Alpha.MOs, this.Alpha.Energies = sort_MOs(this.Alpha.MOs, this.Alpha.Energies, alpha_p_vector) 
    this.Beta.MOs, this.Beta.Energies = sort_MOs(this.Beta.MOs, this.Beta.Energies, beta_p_vector) 

def make_p_vector(new_MOs, ref_MOs, AO_overlaps):
    MO_overlaps = ref_MOs.T.dot(AO_overlaps).dot(new_MOs)
    p_vector = [sum(col) for col in MO_overlaps.T]
    return np.abs(p_vector)

# See if this can be rewriten better 
def sort_MOs(MOs, energies, p):
    """Sorts MOs and energies in decending order
    based on a vector p (the overlap vector)"""
    temp = [[p[i],MOs[:,i],energies[i]] for i in range(len(p))]
    temp = sorted(temp, key = lambda pair: pair[0], reverse = True)     #sorts the elements on the basis of the p values (temp[0])
    new_MOs = np.array([line[1] for line in temp]).T
    new_energies = np.array([line[2] for line in temp])

    return new_MOs, new_energies

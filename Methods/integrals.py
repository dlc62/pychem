# System libraries
import sys
import numpy
from copy import copy
import itertools as it
# C modules
import _c_ints
# Python modules
from Data import constants as c

if sys.version_info.major is 2:
    zip = it.izip

# =================================================================================== #
#  STRUCTURES REQUIRED FOR ONE- AND TWO-ELECTRON INTEGRAL EVALUATION                  #
# =================================================================================== #
class SetRR1:
    def __init__(self,la,lb,inttype):

       i0 = 0; i1 = 1 

       # Set final integral class, and work backwards through RRs to generate terms
       HRR = [[la,lb]]; VRR = []
       HRR_terms = []; VRR_terms = []

       # Generate unique HRR terms
       index = 0
       while index < len(HRR):
          if HRR[index][i1] != 0:
             generate_HRR_terms(HRR,index,0,1,unique=True)
             HRR_terms.append(HRR[index])
          else:
             VRR.append([HRR[index],0])
          index += 1
       HRR_target = list(reversed(HRR_terms))

       # Generate unique VRR terms
       index = 0
       while index < len(VRR):
          if VRR[index][0][i0] != 0:
             generate_VRR_terms(VRR,index,0,1,inttype,unique=True)
             VRR_terms.append(VRR[index])
          index += 1
       VRR_target = list(reversed(VRR_terms))

       # Generate HRR base integral classes
       HRR_base = [];
       for index in range(0,len(HRR_target)):
          HRR_base.append(generate_HRR_terms(HRR_target,index,0,1,unique=False))

       # Generate VRR base integral classes
       VRR_base = [];
       for index in range(0,len(VRR_target)): 
          VRR_base.append(generate_VRR_terms(VRR_target,index,0,1,inttype,unique=False))

       # Store in 'order' object 
       # _base is list containing lists of all base integral classes required for each target integral class.
       #  Integral classes are themselves encoded as lists of angular momentum quantum numbers
       # _target is list of containing all target integral classes (encoded as described above)
       # Remove auxiliary indices for overlap integrals
       if inttype == 'overlap':
          VRR_target_overlap = [VRR_term[0] for VRR_term in VRR_target]
          VRR_base_overlap = []
          for VRR_terms in VRR_base: 
             VRR_base_overlap.append([VRR_term[0] for VRR_term in VRR_terms])
          self.VRR_target = VRR_target_overlap; self.VRR_base = VRR_base_overlap
       else:
          self.VRR_target = VRR_target; self.VRR_base = VRR_base
       self.HRR_target = HRR_target; self.HRR_base = HRR_base

# ----------------------------------------------------------------------------------- #
class SetRR2:
    def __init__(self,la,lb,lc,ld):

       # Use variable indices in case order of operations needs to change
#       angmom = [la,lb,lc,ld]
       i0 = 0; i1 = 1; i2 = 2; i3 = 3
       self.Goofy_bra = False; self.Goofy_ket = False
       if lb > la:
          i0 = 1; i1 = 0
          self.Goofy_bra = True
       if ld > lc:
          i2 = 3; i3 = 2
          self.Goofy_ket = True
       self.VRR1_indices = [i0,i1,i2,i3]
       self.VRR2_indices = [i2,i3,i0,i1]
       self.HRR1_indices = [i1,i0,i2,i3]
       self.HRR2_indices = [i3,i2,i0,i1]

       # Set final integral class, and work backwards through RRs to generate terms
       HRR = [[la,lb,lc,ld]]; VRR = []
       HRR1_terms = []; HRR2_terms = []
       VRR1_terms = []; VRR2_terms = []

       # Generate unique HRR terms, initial HRR term is highest VRR term
       index = 0
       while index < len(HRR):
          if HRR[index][i1] != 0:
             generate_HRR_terms(HRR,index,i1-1,i1)
             HRR1_terms.append(HRR[index])
          elif HRR[index][i3] != 0:
             generate_HRR_terms(HRR,index,i3-1,i3)
             HRR2_terms.append(HRR[index])
          else:
             VRR.append([HRR[index],0])
          index += 1
       HRR1_target = list(reversed(HRR1_terms))
       HRR2_target = list(reversed(HRR2_terms))

       # Generate unique VRR terms, terminates as [00|00]
       index = 0
       while index < len(VRR):
          if VRR[index][0][i0] != 0:
             generate_VRR_terms(VRR,index,i0,i2)
             VRR1_terms.append(VRR[index])
          elif VRR[index][0][i2] != 0:
             generate_VRR_terms(VRR,index,i2,i0)
             VRR2_terms.append(VRR[index])
          index += 1
       VRR1_target = list(reversed(VRR1_terms))
       VRR2_target = list(reversed(VRR2_terms))

       # Generate HRR base integral classes
       HRR1_base = []; HRR2_base = []
       for index in range(0,len(HRR1_target)):
          HRR1_base.append(generate_HRR_terms(HRR1_target,index,i1-1,i1,unique=False))
       for index in range(0,len(HRR2_target)):
          HRR2_base.append(generate_HRR_terms(HRR2_target,index,i3-1,i3,unique=False))

       # Generate VRR base integral classes
       VRR1_base = []; VRR2_base = []
       for index in range(0,len(VRR1_target)): 
          VRR1_base.append(generate_VRR_terms(VRR1_target,index,i0,i2,unique=False))
       for index in range(0,len(VRR2_target)): 
          VRR2_base.append(generate_VRR_terms(VRR2_target,index,i2,i0,unique=False))

       # Store in 'order' object
       # _base is list of lists of tuples
       # _target is list of lists
       self.VRR1_target = VRR1_target; self.VRR1_base = VRR1_base
       self.VRR2_target = VRR2_target; self.VRR2_base = VRR2_base
       self.HRR1_target = HRR1_target; self.HRR1_base = HRR1_base
       self.HRR2_target = HRR2_target; self.HRR2_base = HRR2_base

# ----------------------------------------------------------------------------------- #
def generate_HRR_terms(HRR,index,i0,i1,unique=True):
   t1 = HRR[index][:]
   t1[i1] -= 1
   t0 = t1[:]
   t0[i0] += 1
   terms = [t0,t1]
   if (unique):
      for term in terms:
         if term not in HRR:
            HRR.append(term)
   else:
      return terms

def generate_VRR_terms(VRR,index,i0,i2,inttype=None,unique=True):
   t0 = VRR[index][0][:]; m0 = VRR[index][1]; m1 = m0+1
   t0[i0] -= 1
   t1 = t0[:]
   t1[i0] -= 1
   t2 = t0[:]
   t2[i2] -= 1
   terms = []
   if inttype is 'overlap':
      if t0[i0] > -1:
         terms.append([t0,0])
      if t1[i0] > -1:
         terms.append([t1,0])
   else:
      if t0[i0] > -1:
         terms.append([t0,m0])
         terms.append([t0,m1])
      if t1[i0] > -1:
         terms.append([t1,m0])
         terms.append([t1,m1])
      if t2[i2] > -1:
         terms.append([t2,m1])
   if (unique):
      for term in terms:
         if term not in VRR:
            VRR.append(term)
   else:
      return terms

# =================================================================================== #
#  NUCLEAR REPULSION ENERGY                                                           #
# =================================================================================== #
def nuclear_repulsion(molecule):
    nuclear_repulsion_energy = 0.0e0
    for (i,atom1) in enumerate(molecule.Atoms):
       for (j,atom2) in enumerate(molecule.Atoms):
          if j < i:
             r = distance(atom1.Coordinates,atom2.Coordinates)
             Z1 = atom1.NuclearCharge
             Z2 = atom2.NuclearCharge
             nuclear_repulsion_energy += Z1*Z2/r
          else:
             continue
    return nuclear_repulsion_energy

def distance(v1,v2):
    [x1,y1,z1] = v1
    [x2,y2,z2] = v2
    r = ((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**0.5
    return r

#@profile
# ==================================================================================== #
#  ONE-ELECTRON INTEGRALS: OVERLAP, (KINETIC, NUCLEAR ATTRACTION) -> CORE FOCK         #
# ==================================================================================== #
def one_electron(molecule,shell_pair):

    # -------------------------------------------------------------------------------- #
    #  Unpack shell-pair values and quantity arrays from shell_pair objects            #
    # -------------------------------------------------------------------------------- #
    l_max = shell_pair.Ltot
    m_max = l_max+1

    # -------------------------------------------------------------------------------- #
    #  Swap centres 1 and 2 if lB > lA                                                 #
    # -------------------------------------------------------------------------------- #
    A = shell_pair.Centre1; B = shell_pair.Centre2 
    rAB = shell_pair.CentreDisplacement 
    sigma_P = shell_pair.PrimitivePairSigmas
    U_P = shell_pair.PrimitivePairOverlaps
    P = shell_pair.PrimitivePairCentres
    cc_P = shell_pair.ContractionCoeffs
    nm_P = shell_pair.Normalization
    Goofy = False
    if B.Cgtf.AngularMomentum > A.Cgtf.AngularMomentum:
      A, B = B, A
      rAB = -rAB
      sigma_P = sigma_P.transpose((1,0))
      U_P = U_P.transpose((1,0))
      P = P.transpose((1,0,2))
      cc_P = cc_P.transpose((1,0))
      nm_P = nm_P.transpose((1,0))
      Goofy = True

    lA = A.Cgtf.AngularMomentum; nlA = c.nAngMomFunctions[lA]; nA = A.Cgtf.NPrimitives; rA = A.Coords 
    lB = B.Cgtf.AngularMomentum; nlB = c.nAngMomFunctions[lB]; nB = B.Cgtf.NPrimitives; rB = B.Coords
    nm_P = nm_P.reshape((nlA,nlB)); exB = B.Cgtf.Exponents 

    # -------------------------------------------------------------------------------- #
    #  Set up structures required for integral evaluation                              #
    # -------------------------------------------------------------------------------- #
    order_nuclear = SetRR1(lA,lB,'nuclear')
    order_overlap = SetRR1(lA,lB+2,'overlap')
    overlap_integrals = {}; nuclear_integrals = {}

    # -------------------------------------------------------------------------------- #
    #  Compute and store primitive fundamentals                                        #
    # -------------------------------------------------------------------------------- #

    # Nuclear attraction
    nuclear_fundamentals = numpy.zeros((m_max,nA,nB),dtype="double")
    for (iatom,atom) in enumerate(molecule.Atoms):
       rC = atom.Coordinates
       Z = atom.NuclearCharge
#_C_INTS
       _c_ints.one_electron_fundamentals(nuclear_fundamentals, sigma_P, U_P, P, rC, Z, nA, nB, l_max) 
#_C_INTS
       for m in range(0,m_max):
          nuclear_integrals[(iatom,0,0,m)] = copy(nuclear_fundamentals[m])
#          print('c nuclear_fundamentals:', nuclear_fundamentals[m])
       nuclear_fundamentals.fill(0.0)

    # Overlap
    overlap_integrals[(0,0)] = shell_pair.PrimitivePairOverlaps

    # -------------------------------------------------------------------------------- #
    #  Apply VRR to form uncontracted [m|Vn|0] and [m+2||0] integrals                  #
    # -------------------------------------------------------------------------------- #

    # Nuclear attraction
    if order_nuclear.VRR_target != []:
       for (iatom,atom) in enumerate(molecule.Atoms):
          rC = atom.Coordinates
          Z = atom.NuclearCharge
          do_1e_vrr(order_nuclear, nuclear_integrals, sigma_P, P, rA, rC, nA, nB, iatom)

    # Overlap
    do_1e_vrr(order_overlap, overlap_integrals, sigma_P, P, rA, [], nA, nB, -1)

    # -------------------------------------------------------------------------------- #
    #  Apply HRR to form uncontracted [m|Vn|n] and [m||n+2] integrals                  #
    # -------------------------------------------------------------------------------- #

    # Nuclear attraction
    for iatom in range(0,molecule.NAtom):
       do_1e_hrr(order_nuclear, nuclear_integrals, rAB, nA, nB, iatom)

    # Overlap
    do_1e_hrr(order_overlap, overlap_integrals, rAB, nA, nB, -1)

    # -------------------------------------------------------------------------------- #
    #  Use derivative relation to form uncontracted [m|T|n] integrals directly         #
    # -------------------------------------------------------------------------------- #

    kinetic_ints = numpy.zeros((nlA*nA,nlB*nB), dtype="double") 
    try:
       base_ints = [copy(overlap_integrals[(lA,lB+2)]), copy(overlap_integrals[(lA,lB)]),copy(overlap_integrals[(lA,lB-2)])]
    except:
       base_ints = [copy(overlap_integrals[(lA,lB+2)]), copy(overlap_integrals[(lA,lB)])]
#_C_INTS
    _c_ints.one_electron_kinetic(kinetic_ints, base_ints, exB[:], len(base_ints), nA, nB, lA, lB)
#_C_INTS

    # -------------------------------------------------------------------------------- #
    #  Compute contracted (m|O|n) integrals                                            #
    # -------------------------------------------------------------------------------- #

    # Nuclear attraction
    summed_contracted_nuclear_ints = numpy.zeros((nlA,nlB), dtype="double")
    for iatom in range(0,molecule.NAtom):
       contracted_nuclear_ints = numpy.zeros((nlA,nlB), dtype="double")
       nuclear_ints = copy(nuclear_integrals[(iatom,lA,lB,0)])
#_C_INTS
       _c_ints.one_electron_contract(contracted_nuclear_ints, nuclear_ints, cc_P[:,:], nA, nB, lA, lB)
#_C_INTS
       summed_contracted_nuclear_ints += contracted_nuclear_ints

    # Kinetic and overlap (kinetic_ints from above)
    overlap_ints = copy(overlap_integrals[(lA,lB)])
    contracted_kinetic_ints = numpy.zeros((nlA,nlB), dtype="double")
    contracted_overlap_ints = numpy.zeros((nlA,nlB), dtype="double")
#_C_INTS
    _c_ints.one_electron_contract(contracted_kinetic_ints, kinetic_ints, cc_P[:,:], nA, nB, lA, lB)
    _c_ints.one_electron_contract(contracted_overlap_ints, overlap_ints, cc_P[:,:], nA, nB, lA, lB)
#_C_INTS

    # -------------------------------------------------------------------------------- #
    #  Extract final target contracted (m|n) integrals                                 #
    # -------------------------------------------------------------------------------- #
    
    contracted_overlap = numpy.multiply(contracted_overlap_ints,nm_P)
    contracted_nuclear = numpy.multiply(summed_contracted_nuclear_ints,nm_P)
    contracted_kinetic = numpy.multiply(contracted_kinetic_ints,nm_P)
    contracted_core = numpy.multiply(summed_contracted_nuclear_ints + contracted_kinetic_ints,nm_P)

    if Goofy:
#       return contracted_kinetic.T, contracted_nuclear.T, contracted_overlap.T
       return contracted_core.T, contracted_overlap.T
    else:
#       return contracted_kinetic, contracted_nuclear, contracted_overlap
       return contracted_core, contracted_overlap

# =================================================================================== #
# iatom acts as flag to decide on whether to compute overlap integrals (iatom = -1) or nuclear attraction (otherwise)
# =================================================================================== #
def do_1e_vrr(order, integrals, sigma_P, P, rA, rC, nA, nB, iatom):

   for (target_class, base_classes) in zip(order.VRR_target, order.VRR_base):

      if iatom == -1:
         lA = target_class[0]
         base_ints = [copy(integrals[tuple(base_class)]) for base_class in base_classes]
      else:
         lA = target_class[0][0]
         base_ints = [copy(integrals[tuple([iatom]+base_class[0]+[base_class[1]])]) for base_class in base_classes]

      nlA = c.nAngMomFunctions[lA]
      target_ints = numpy.zeros((nlA*nA,nB), dtype="double")
#_C_INTS    
      _c_ints.one_electron_vrr(target_ints, base_ints, sigma_P[:,:], P[:,:,:], rA[:], rC[:], iatom, len(base_ints), nA, nB, lA)
#_C_INTS

      if iatom == -1:
         integrals[tuple(target_class)] = copy(target_ints)
      else:
         integrals[tuple([iatom]+target_class[0]+[target_class[1]])] = copy(target_ints)

# ----------------------------------------------------------------------------------- #
def do_1e_hrr(order, integrals, rAB, nA, nB, iatom):

   for (target_class, base_classes) in zip(order.HRR_target, order.HRR_base):

      if iatom == -1:
         base_ints = [copy(integrals[tuple(base_class)]) for base_class in base_classes]
      else:
         base_ints = [copy(integrals[tuple([iatom]+base_class+[0])]) for base_class in base_classes]
      
      lA = target_class[0]; lB = target_class[1]
      nlA = c.nAngMomFunctions[lA]; nlB = c.nAngMomFunctions[lB]
      target_ints = numpy.zeros((nlA*nA,nlB*nB), dtype="double")

#_C_INTS
      _c_ints.one_electron_hrr(target_ints, base_ints, rAB[:], len(base_ints), nA, nB, lA, lB)
#_C_INTS

      if iatom == -1:
         integrals[tuple(target_class)] = copy(target_ints)
      else:
         integrals[tuple([iatom]+target_class+[0])] = copy(target_ints)

# =================================================================================== #

#@profile
# ==================================================================================== #
#  TWO-ELECTRON REPULSION INTEGRALS - HGP ALGORITHM WITH SCHWARZ BOUND SCREENING       #
# ==================================================================================== #
def two_electron(shell_pair1,shell_pair2,ints_type,grid_value):

#    print("c_integrals")
    # -------------------------------------------------------------------------------- #
    #  Swap bra and ket if lket > lbra                                                 #
    # -------------------------------------------------------------------------------- #
    Goofy = False
    if shell_pair2.Ltot > shell_pair1.Ltot:
       shell_pair1,shell_pair2 = shell_pair2,shell_pair1 
       Goofy = True

    # -------------------------------------------------------------------------------- #
    #  Extract atom-centric data from shell-pair objects                               #
    # -------------------------------------------------------------------------------- #
    A = shell_pair1.Centre1; B = shell_pair1.Centre2
    C = shell_pair2.Centre1; D = shell_pair2.Centre2
    lA = A.Cgtf.AngularMomentum; nA = A.Cgtf.NPrimitives; nlA = A.Cgtf.NAngMom
    lB = B.Cgtf.AngularMomentum; nB = B.Cgtf.NPrimitives; nlB = B.Cgtf.NAngMom
    lC = C.Cgtf.AngularMomentum; nC = C.Cgtf.NPrimitives; nlC = C.Cgtf.NAngMom
    lD = D.Cgtf.AngularMomentum; nD = D.Cgtf.NPrimitives; nlD = D.Cgtf.NAngMom

#    # -------------------------------------------------------------------------------- #
#    #  Evaluate contracted integral upper bounds, skip this shell-quartet entirely     #
#    #  and return None if all contracted integrals in this class are negligible        #
#    #  Skip if the entire aim is to evaluate bound integrals, replacing C_P = C_Q = [] #
#    # -------------------------------------------------------------------------------- #
#    C_P = shell_pair1.ContractedBoundInts; C_Q = shell_pair2.ContractedBoundInts
#    if (C_P is None) or (C_Q is None):
#       return None
#    elif (C_P != []):
##_C_INTS
#       _c_ints.two_electron_bound(bound, C_P, C_Q, nlA, nlB, nlC, nlD)
##_C_INTS
#       if numpy.amax(bound) < c.integral_threshold:
#          return None

    # -------------------------------------------------------------------------------- #
    #  Allocate numpy array to store fundamental electron repulsion integrals          #
    #  and centres for current shell-quartet                                           #
    # -------------------------------------------------------------------------------- #
    lBra = shell_pair1.Ltot; lKet = shell_pair2.Ltot
    l_max = lBra+lKet
    m_max = l_max+1
    fundamentals = numpy.zeros((m_max,nA*nB,nC*nD), dtype="double") 
    R = numpy.zeros((nA*nB,nC*nD,3), dtype="double")

    # -------------------------------------------------------------------------------- #
    #  Unpack bra/ket data (shell-pair quantity arrays) from shell_pair objects        #
    # -------------------------------------------------------------------------------- #
    sigma_P = shell_pair1.PrimitivePairSigmas; sigma_Q = shell_pair2.PrimitivePairSigmas
    U_P = shell_pair1.PrimitivePairOverlaps; U_Q = shell_pair2.PrimitivePairOverlaps
    P = shell_pair1.PrimitivePairCentres; Q = shell_pair2.PrimitivePairCentres
    zeta_P = shell_pair1.PrimitivePairHalfSigmas; zeta_Q = shell_pair2.PrimitivePairHalfSigmas
    R_P = shell_pair1.CentreDisplacement; R_Q = shell_pair2.CentreDisplacement
    cc_P = shell_pair1.ContractionCoeffs; cc_Q = shell_pair2.ContractionCoeffs
    nm_P = shell_pair1.Normalization; nm_Q = shell_pair2.Normalization

    # -------------------------------------------------------------------------------- #
    #  Compute fundamentals                                                            #
    #  ints_type: 0 = electron repulsion, 1 = scattering                               #
    # -------------------------------------------------------------------------------- #
#_C_INTS
    _c_ints.two_electron_fundamentals(fundamentals, sigma_P[:,:], U_P[:,:], P[:,:,:], 
                                      sigma_Q[:,:], U_Q[:,:], Q[:,:], R[:,:,:],
                                      nA, nB, nC, nD, l_max, ints_type, grid_value)
#_C_INTS

    # -------------------------------------------------------------------------------- #
    #  Set up global order class that controls all RR ordering                         #
    # -------------------------------------------------------------------------------- #
    order = SetRR2(lA,lB,lC,lD)
    
    # -------------------------------------------------------------------------------- #
    #  Set up structures required to calculate and store primitive integrals           #
    # -------------------------------------------------------------------------------- #
    kappa_P = B.Cgtf.DoubleExponents
    kappa_Q = D.Cgtf.DoubleExponents
    if order.Goofy_bra: kappa_P = A.Cgtf.DoubleExponents
    if order.Goofy_ket: kappa_Q = C.Cgtf.DoubleExponents

    integrals = {}
    for m in range(0,m_max):
       integrals[(0,0,0,0,m)] = fundamentals[m,:].T
#       print(m,fundamentals[m,:])

    # -------------------------------------------------------------------------------- #
    #  Use VRR to compute uncontracted [m0|n0] 2e- integrals for current shell-quartet #
    # -------------------------------------------------------------------------------- #
    do_2e_vrr(2,order,integrals,[nC,nD,nA,nB],zeta_Q,zeta_P,kappa_Q,R_Q,R.swapaxes(0,1))
    for m in range(0,m_max):
       integrals[(0,0,0,0,m)] = integrals[(0,0,0,0,m)].T
    do_2e_vrr(1,order,integrals,[nA,nB,nC,nD],zeta_P,zeta_Q,kappa_P,R_P,R)

    # -------------------------------------------------------------------------------- #
    #  Compute contracted (m0|n0) two-electron integrals for current shell-quartet     #
    #  Create contracted integrals directory for use in the next HRR steps             #
    #  Note that bra carries the highest angular momentum, determined dynamically      #
    # -------------------------------------------------------------------------------- #
    contracted_integrals = {}
    do_2e_contract(order,contracted_integrals,integrals,[nA,nB,nC,nD],cc_P,cc_Q)
#    print('contracted integrals',contracted_integrals[tuple(order.VRR1_target[-1][0])])

    # -------------------------------------------------------------------------------- #
    #  Use HRR to compute contracted (mn|ls) 2e- integrals for current shell-quartet   #
    #  Do exponent-dependent part of normalization in the process
    # -------------------------------------------------------------------------------- #
    do_2e_hrr(2,order,contracted_integrals,R_Q)
    do_2e_hrr(1,order,contracted_integrals,R_P)

    # -------------------------------------------------------------------------------- #
    #  Finish normalizing contracted basis functions - angular momentum dependent part #
    # -------------------------------------------------------------------------------- #
    normalization = (nm_P.T).dot(nm_Q)
    normalized_contracted_integrals = numpy.multiply(contracted_integrals[(lA,lB,lC,lD)],normalization)
     
    # -------------------------------------------------------------------------------- #
    #  Return contracted integrals                                                     # 
    # -------------------------------------------------------------------------------- #
    # reshape to 4D array?
    if Goofy:
       return normalized_contracted_integrals.T.reshape(nlC,nlD,nlA,nlB)
    else:
       return normalized_contracted_integrals.reshape(nlA,nlB,nlC,nlD)

# =================================================================================== #
def do_2e_vrr(step,order,integrals,n_primitives,zeta,eta,kappa,Rx,R):

    [na,nb,nc,nd] = n_primitives
    kappa_index = 1; sign_Rx = -1
    if step == 1:
       targets = order.VRR1_target; bases = order.VRR1_base; sign_R = -1
       primary_index = order.VRR1_indices[0]; secondary_index = order.VRR1_indices[2]
       if order.Goofy_bra: kappa_index = 0; sign_Rx = 1
    if step == 2:
       targets = order.VRR2_target; bases = order.VRR2_base; sign_R = 1
       primary_index = order.VRR2_indices[0]; secondary_index = order.VRR2_indices[2]
       if order.Goofy_ket: kappa_index = 0; sign_Rx = 1

    for (target_class, base_classes) in zip(targets, bases):
       tc_key = tuple(target_class[0]+[target_class[1]])
       base_ints = [copy(integrals[tuple(bc[0]+[bc[1]])]) for bc in base_classes]
       lbra = target_class[0][primary_index]; nl_bra = c.nAngMomFunctions[lbra]
       lket = target_class[0][secondary_index]; nl_ket = c.nAngMomFunctions[lket]
       target_ints = numpy.zeros((nl_bra*na*nb,nl_ket*nc*nd), dtype="double") 
#_C_INTS
       _c_ints.two_electron_vrr(target_ints, base_ints, zeta[:,:], eta[:,:], kappa[:], sign_Rx*Rx[:], 
                                sign_R*R[:,:,:], len(base_ints), na, nb, nc, nd, lbra, lket, kappa_index)
#_C_INTS
       integrals[tc_key] = copy(target_ints)

    if step == 2:
       for tc in targets:
          tc_key = tuple(tc[0]+[tc[1]])
          integrals[tc_key] = integrals[tc_key].T
#    if step == 1:
#       tc = targets[-1]
#       tc_key = tuple(tc[0]+[tc[1]])
#       print(tc_key)
#       print(integrals[tc_key])
       
# ----------------------------------------------------------------------------------- #
def do_2e_contract(order,contracted_integrals,integrals,n_primitives,cc_bra,cc_ket):

    if order.VRR1_target == []:
       if order.VRR2_target != []:
          order.VRR1_target.append(VRR2_target[-1])
       else:
          order.VRR1_target.append([[0,0,0,0],0])

    [na,nb,nc,nd] = n_primitives

    for target_class in order.VRR1_target:
       if target_class[-1] == 0:
          tc_key_prim = tuple(target_class[0]+[0])
          tc_key_cont = tuple(target_class[0])
          lbra = target_class[0][order.VRR1_indices[0]]
          lket = target_class[0][order.VRR1_indices[2]]
          contracted_ints = numpy.zeros((c.nAngMomFunctions[lbra],c.nAngMomFunctions[lket]), dtype="double") 
#_C_INTS
          _c_ints.two_electron_contract(contracted_ints, copy(integrals[tc_key_prim]),
                                        cc_bra[:,:], cc_ket[:,:], na, nb, nc, nd, lbra, lket)
#_C_INTS
          contracted_integrals[tc_key_cont] = contracted_ints

# ----------------------------------------------------------------------------------- #
def do_2e_hrr(step,order,integrals,Rx):

    goofy = 0; la_off = 0; lb_off = 1; sign_Rx = 1;
    if step == 1:
       targets = order.HRR1_target; bases = order.HRR1_base
       if order.Goofy_bra: goofy = 1; la_off = 1; lb_off = 0; sign_Rx = -1
    if step == 2:
       targets = order.HRR2_target; bases = order.HRR2_base
       if order.Goofy_ket: goofy = 1; la_off = 1; lb_off = 0; sign_Rx = -1

    for (target_class, base_classes) in zip(targets,bases):
       if step == 1: [la,lb,lc,ld] = base_classes[1] 
       if step == 2: [lc,ld,la,lb] = base_classes[1]
       nla = c.nAngMomFunctions[la+la_off]; nlb = c.nAngMomFunctions[lb+lb_off]
       nlc = c.nAngMomFunctions[lc]; nld = c.nAngMomFunctions[ld]
       target_ints = numpy.zeros((nla*nlb,nlc*nld), dtype="double")
       base_ints = [copy(integrals[tuple(base_class)]) for base_class in base_classes]
#_C_INTS
       if step == 2: 
          _c_ints.two_electron_hrr(target_ints, base_ints[0].T, base_ints[1].T, sign_Rx*Rx[:], la, lb, lc, ld, goofy)
       if step == 1:  
          _c_ints.two_electron_hrr(target_ints, base_ints[0], base_ints[1], sign_Rx*Rx[:], la, lb, lc, ld, goofy)
#_C_INTS
       integrals[tuple(target_class)] = copy(target_ints)

    if step == 2: 
       for target_class in targets:
          integrals[tuple(target_class)] = integrals[tuple(target_class)].T

# =================================================================================== #

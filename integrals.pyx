#!/usr/bin/python

import sys
from math import pi
from math import exp
import constants as c
import copy
import itertools
from scipy.special import gamma
from scipy.special import gammainc
from scipy.misc import factorial

cdef extern from "math.h":
    double tgamma(double)

cdef extern from "math.h":
    double sqrt(double)

# ================================================================================================ #
#  STRUCTURES REQUIRED FOR ONE-ELECTRON INTEGRAL EVALUATION                                        #
# ================================================================================================ #

class SetRR1:
   #generates all the integrals to evaluate
    def __init__(self,l1,l2):
       HRR = [[l1,l2]]; VRR = []
       HRR_terms = []; VRR_terms = []
       index = 0
       while index < len(HRR):                      # loop stops once all HRR terms have been reduced  
          if HRR[index][1] != 0:                    
             generate_HRR_terms(HRR,index,0,1,True) # reduces a single term by one step and add new terms to the list
             HRR_terms.append(HRR[index])           
          else:
             VRR.append([HRR[index],0])             # adds terms to the VRR once they cannot be reduced further via HRR 
          index += 1
       index = 0
       # only terms with s in the ket are added to the VRR 
       while index < len(VRR):
          if VRR[index][0][0] != 0:                 # do nothing if there is an s in the bra
             generate_VRR_terms(VRR,index,0,1)
             VRR_terms.append(VRR[index])
          index += 1
    # storing lists of all required integrals 
    # running from most reduced to least 
       self.hrr = list(reversed(HRR_terms))
       self.vrr = list(reversed(VRR_terms))

# ================================================================================================ #
#  STRUCTURES REQUIRED FOR TWO-ELECTRON INTEGRAL EVALUATION                                        #
# ================================================================================================ #

class ShellQuartet:
    def __init__(self,centreA,centreB,centreC,centreD,primA,primB,primC,primD):
        #the exponent and contraction coefficent respectivly
       [self.alpha,self.cA] = primA
       [self.beta,self.cB] = primB
       [self.gamma,self.cC] = primC
       [self.delta,self.cD] = primD
       #nuclear corrdinates
       self.rA = centreA.Coords
       self.rB = centreB.Coords
       self.rC = centreC.Coords
       self.rD = centreD.Coords
       #angular momentum
       self.lA = centreA.Cgtf.AngularMomentum
       self.lB = centreB.Cgtf.AngularMomentum
       self.lC = centreC.Cgtf.AngularMomentum
       self.lD = centreD.Cgtf.AngularMomentum
       #constatans needed for integral evaluation 
       self.zeta = self.alpha+self.beta
       self.eta = self.gamma+self.delta
       self.Gab = exp(-self.alpha*self.beta/self.zeta*distance(self.rA,self.rB))
       self.Gcd = exp(-self.gamma*self.delta/self.eta*distance(self.rC,self.rD))
       self.Pab = [(self.alpha*a+self.beta*b)/self.zeta for (a,b) in itertools.izip(self.rA,self.rB)]
       self.Pcd = [(self.gamma*c+self.delta*d)/self.eta for (c,d) in itertools.izip(self.rC,self.rD)]
       self.tau = self.zeta+self.eta
       self.rho = self.zeta*self.eta/self.tau
       self.T = self.rho*distance(self.Pab,self.Pcd)
       self.W = [(self.zeta*pab+self.eta*pcd)/self.tau for (pab,pcd) in itertools.izip(self.Pab,self.Pcd)] 

class ReIndex:
    # Re-indexes shell-quartet (sq) data for efficient RR generation,
    # so bra carries highest angular momentum and i1,i3 point to centres
    # of highest angular momentum in bra and ket, respectively
    def __init__(self,sq):
       self.W = sq.W; self.rho = sq.rho; self.tau = sq.tau
       self.i1 = 0 ; self.i2 = 1 ; self.i3 = 2 ; self.i4 = 3
       self.l1 = sq.lA; self.l2 = sq.lB; self.l3 = sq.lC; self.l4 = sq.lD
       self.r1 = sq.rA; self.r2 = sq.rB; self.r3 = sq.rC; self.r4 = sq.rD
       #putting the larger angular momentum in the bra on the right
       if sq.lB > sq.lA:
          self.i1,self.i2 = self.i2,self.i1
          self.l1,self.l2 = self.l2,self.l1
          self.r1,self.r2 = self.r2,self.r1
       #same for the ket 
       if sq.lD > sq.lC:
          self.i3,self.i4 = self.i4,self.i3
          self.l3,self.l4 = self.l4,self.l3
          self.r3,self.r4 = self.r4,self.r3
       #putting the largest total angular momentum in the bra 
       if (self.l3+self.l4) > (self.l1+self.l2):
          self.i1,self.i2,self.i3,self.i4 = self.i3,self.i4,self.i1,self.i2
          self.l1,self.l2,self.l3,self.l4 = self.l3,self.l4,self.l1,self.l2
          self.r1,self.r2,self.r3,self.r4 = self.r3,self.r4,self.r1,self.r2
          self.P = sq.Pcd; self.Q = sq.Pab; self.z = sq.eta; self.e = sq.zeta
       else:
           # ?? Why is this in an else block ?? 
          self.P = sq.Pab; self.Q = sq.Pcd; self.z = sq.zeta; self.e = sq.eta 
       self.ltot = self.l1 + self.l2 + self.l3 + self.l4
     
class SetRR:
    # Generate all the reduced integral classes to be calculated for a single starting ERI
    def __init__(self,bk):
      #HRR contains the angular momentum numbers 
       HRR = [[bk.l1,bk.l2,bk.l3,bk.l4]]; VRR = []
       HRR_bra_terms = []; HRR_ket_terms = [];
       VRR_bra_terms = []; VRR_ket_terms = [];
       index = 0
      #reducing the classes to (a0|b0) using the HRR
       while index < len(HRR):
          if HRR[index][1] != 0:
             generate_HRR_terms(HRR,index,0,1)
             HRR_bra_terms.append(HRR[index])
          elif HRR[index][3] != 0:
             generate_HRR_terms(HRR,index,2,3)
             HRR_ket_terms.append(HRR[index])
          else:
            #if integral is of form (a0|c0) it falls through to the the VRR list 
            #HRR doesn't grow longer so index can become greater than len(HRR)
             VRR.append([HRR[index],0])
          index += 1
       index = 0
      #applying the VRR
       while index < len(VRR):
          if VRR[index][0][0] != 0:
             generate_VRR_terms(VRR,index,0,2)
             VRR_bra_terms.append(VRR[index])
          elif VRR[index][0][2] != 0:
             generate_VRR_terms(VRR,index,2,0)
             VRR_ket_terms.append(VRR[index])
          index += 1
       self.hrr_bra = list(reversed(HRR_bra_terms))
       self.hrr_ket = list(reversed(HRR_ket_terms))
       self.vrr_bra = list(reversed(VRR_bra_terms))
       self.vrr_ket = list(reversed(VRR_ket_terms))

# HRR :: [[Int, Int]] for one electron integrals 
def generate_HRR_terms(HRR,index,i0,i1,kinetic=False):
  # HRR[index] => t0, t1 
  # (a(b+1)|cd) => ((a+1)b|cd),(ab|cd)
   t1 = HRR[index][:]     # pick a term to reduce  
   t1[i1] -= 1            
   t0 = t1[:]             
   t0[i0] += 1            # make a second new term by decrementing another  
   terms = [t0,t1]
   if kinetic:            # if kinetic make a third term 
      t2 = t1[:]
      t2[i0] -= 1
      terms.append(t2)
   for term in terms:
      if term not in HRR:
        #add the new terms to the list produced via the HRR
         HRR.append(term)

def generate_VRR_terms(VRR,index,i0,i2):
  #VRR[index] => t0^m, t0^(m+1), t1^(m+1), t2^(m+1)
  #((a+1)0|c0) => (a0|c0)    ^m,    (a0|c0)^(m+1),
  #               ((a-1)0|c0)^m,    ((a-1)0|c0)^(m+1),
  #                                 (a0|(c-1)0)^(m+1)
   t0 = VRR[index][0][:]; m0 = VRR[index][1]; m1 = m0+1 
   t0[i0] -= 1
   t1 = t0[:]
   t1[i0] -= 1
   t2 = t0[:]
   t2[i2] -= 1
   terms = []
  #ifs throw out integrals with less than zero angular momentum. 
   if t0[i0] > -1:
      terms.append([t0,m0])
      terms.append([t0,m1])
   if t1[i0] > -1:
      terms.append([t1,m0])
      terms.append([t1,m1])
   if t2[i2] > -1:
      terms.append([t2,m1])
   for term in terms:
      if term not in VRR:
         VRR.append(term)

# ================================================================================================ #
#  NUCLEAR REPULSION ENERGY                                                                        #
# ================================================================================================ #
def nuclear_repulsion(molecule):
    nuclear_repulsion_energy = 0.0e0
    #i and j from enumerate are the atom indices
    for (i,atom1) in enumerate(molecule.Atoms):
       for (j,atom2) in enumerate(molecule.Atoms):
         #evaluating nuclear repulsion using Coloumb's Law  
          if j < i:
             r = sqrt(distance(atom1.Coordinates,atom2.Coordinates))
             Z1 = atom1.NuclearCharge
             Z2 = atom2.NuclearCharge
             nuclear_repulsion_energy += Z1*Z2/r
          else:
             continue
    return nuclear_repulsion_energy

# ================================================================================================ #
#  ONE-ELECTRON INTEGRALS: KINETIC, NUCLEAR ATTRACTION, OVERLAP -> CORE FOCK MATRIX ELEMENTS       #
# ================================================================================================ #
def one_electron(molecule,shell_pair):
    # VRR follows Obara & Saika, JCP, 84, 1986, 3963-3974
    # HRR from Head-Gordon & Pople, JCP, 89, 1998, 5777-5786
    # Set up indexing and storage for integrals in original order
    CgtfA = shell_pair.Centre1.Cgtf; CgtfB = shell_pair.Centre2.Cgtf
    rA = shell_pair.Centre1.Coords; rB = shell_pair.Centre2.Coords
    lA = CgtfA.AngularMomentum; lB = CgtfB.AngularMomentum; ltot = lA+lB
    nlA = CgtfA.NAngMom; nlB = CgtfB.NAngMom
    row = [0.0 for i in range(0,CgtfB.NAngMom)]
    contracted_core = [copy.deepcopy(row[:]) for i in range(0,CgtfA.NAngMom)]  
    contracted_overlap = copy.deepcopy(contracted_core)
    contracted_kinetic = copy.deepcopy(contracted_core)
    contracted_nuclear = copy.deepcopy(contracted_core)
    # Determine integral construction ordering
    Swap = False
    if lB > lA:
       Swap = True
    if Swap:
   # putting the higher angular momentum function int the bra 
   # and finding all the integrals to evaluate
       i1 = 1; i2 = 0; r1 = rB; r2 = rA; l1 = lB; l2 = lA; order = SetRR1(lB,lA)
    else:
       i1 = 0; i2 = 1; r1 = rA; r2 = rB; l1 = lA; l2 = lB; order = SetRR1(lA,lB)
    # iterating over all combinations of primitives 
    for [alpha,cA] in CgtfA.Primitives:
       for [beta,cB] in CgtfB.Primitives:
          zeta = alpha+beta
          xi = alpha*beta/zeta
          Rsq = distance(rA,rB)
          Gab = exp(-xi*Rsq)
          Pab = [(alpha*A + beta*B)/zeta for (A,B) in itertools.izip(rA,rB)]
       # -------------------------------------------------------------------------------------- #
       #                           Initialize fundamental integrals                             #
       # -------------------------------------------------------------------------------------- #
          overlap = {}
          kinetic = {}
          nuclear_row = [{} for i in range(0,ltot+1)]
          nuclear = [copy.deepcopy(nuclear_row) for atom in molecule.Atoms]
          lkey = (0,0,0,0,0,0) 
         #overlap and kinetic integrals for s functions  
          overlap[lkey] = (pi/zeta)**1.5e0*Gab
          kinetic[lkey] = xi*(3-2*xi*Rsq)*overlap[lkey]
          for (iatom,atom) in enumerate(molecule.Atoms):
             Rc = atom.Coordinates
             Z = atom.NuclearCharge
             U = zeta*distance(Pab,Rc)
             F = F_aux(ltot,U)
            # nuclear integral for s functions  
             for m in range(0,ltot+1):
                nuclear[iatom][m][lkey] = -2*(zeta/pi)**0.5*overlap[lkey]*Z*F[m]   # Generating base nuclear integrals
       # -------------------------------------------------------------------------------------- #
       #                                       VRR step                                         #
       # -------------------------------------------------------------------------------------- #
          z1 = alpha; z2 = beta
          if Swap:                           # put the highest angular momentum function the the bra
             z1 = beta; z2 = alpha
          index1 = 0
          for [term,m] in order.vrr:         # steping over integrals starting with the (s||s) terms  
            #all VRR terms needed for that integral    
             l1 = term[index1];
             lpair = [[],[]]
             lpair[i2] = c.lQuanta[0][0][:]
             for qm in c.lQuanta[l1]:        # qm are angular momentum numbers (m) for each Cartesian component of a given angular momentum, 
                lpair[i1] = qm[:]
                lkey = make_key(lpair)
                for j in range(0,3):         # choose Cartesian component index to decrement 
                   if qm[j] != 0:
                      j1 = j                 # j is Cartesian component index
                lpair[i1][j1] -= 1           # do decrementing in-place, sequentially
                ln1 = lpair[i1][j1]
                lk0 = make_key(lpair)
                lpair[i1][j1] -= 1
                lk1 = make_key(lpair)
                overlap[lkey] =   (Pab[j1]-r1[j1])*get(overlap,lk0) \
                                 + ln1*get(overlap,lk1)/(2*zeta)
                kinetic[lkey] =   (Pab[j1]-r1[j1])*get(kinetic,lk0)+2*xi*get(overlap,lkey) \
                                 + ln1*(get(kinetic,lk1)/(2*zeta)-2*xi*get(overlap,lk1)/(2*z1))
                for (iatom,atom) in enumerate(molecule.Atoms):
                   Rc = atom.Coordinates
                   nuclear[iatom][m][lkey] =   (Pab[j1]-r1[j1])*get(nuclear[iatom][m],lk0) \
                                             - (Pab[j1]-Rc[j1])*get(nuclear[iatom][m+1],lk0) \
                                             + ln1*(get(nuclear[iatom][m],lk1)-get(nuclear[iatom][m+1],lk1))/(2*zeta) 
       # -------------------------------------------------------------------------------------- #
       #                                       HRR step                                         #
       # -------------------------------------------------------------------------------------- #
          index1 = 0; index2 = 1;
          for term in order.hrr:
             l1 = term[index1];
             l2 = term[index2];
             for qm in c.lQuanta[l1]:
                for qn in c.lQuanta[l2]:
                   lpair[i1] = qm[:]
                   lpair[i2] = qn[:]
                   lkey = make_key(lpair)
                   for j in range(0,3):      # choose Cartesian component index to decrement (i2) or increment (i1)
                      if qn[j] != 0:
                         j2 = j 
                   lpair[i2][j2] -= 1
                   lk0 = make_key(lpair)
                   lna = lpair[i1][j2]
                   lnb = lpair[i2][j2]
                   lpair[i1][j2] += 1
                   lk1 = make_key(lpair)
                   lpair[i1][j2] -= 2
                   lka = make_key(lpair)
                   lpair[i1][j2] += 2
                   lpair[i2][j2] -= 1
                   lkb = make_key(lpair)
                   overlap[lkey] = (r1[j2]-r2[j2])*get(overlap,lk0) + get(overlap,lk1)
                   kinetic[lkey] = (r1[j2]-r2[j2])*get(kinetic,lk0) + get(kinetic,lk1) + \
                                   2*xi*((r1[j2]-r2[j2])*get(overlap,lk0) + lna*get(overlap,lka)/(2*z1) - lnb*get(overlap,lkb)/(2*z2))
                   for iatom in range(0,molecule.NAtom):
                      nuclear[iatom][0][lkey] = (r1[j2]-r2[j2])*get(nuclear[iatom][0],lk0) + get(nuclear[iatom][0],lk1)
       # -------------------------------------------------------------------------------------- #
       #                           Contraction step and screening                               #
       # -------------------------------------------------------------------------------------- #
#          normalized_contraction_coeffs = copy.deepcopy(contracted_core)
          for (ilA,qA) in enumerate(c.lQuanta[lA]):
             for (ilB,qB) in enumerate(c.lQuanta[lB]):
                lkey = make_key([qA,qB])
                nA = normalize(alpha,qA)
                nB = normalize(beta,qB)
                cc = nA*nB*cA*cB
#                normalized_contraction_coeffs[ilA][ilB] = cc
                contracted_overlap[ilA][ilB] += cc*overlap[lkey]
                contracted_kinetic[ilA][ilB] += cc*kinetic[lkey]
                for iatom in range(0,molecule.NAtom):
                   contracted_nuclear[ilA][ilB] += cc*nuclear[iatom][0][lkey] 
    for ilA in range(0,nlA):
       for ilB in range(0,nlB):
          contracted_core[ilA][ilB] = contracted_kinetic[ilA][ilB] + contracted_nuclear[ilA][ilB]
    return contracted_core,contracted_overlap


# ================================================================================================ #
#  TWO-ELECTRON REPULSION INTEGRALS                                                                #
# ================================================================================================ #
def two_electron(shell_pair1,shell_pair2):
    s = [0.0e0 for i in range(0,shell_pair2.Centre2.Cgtf.NAngMom)] 
    ls = [copy.deepcopy(s) for i in range(0,shell_pair2.Centre1.Cgtf.NAngMom)]
    nls = [copy.deepcopy(ls) for i in range(0,shell_pair1.Centre2.Cgtf.NAngMom)]
    mnls = [copy.deepcopy(nls) for i in range(0,shell_pair1.Centre1.Cgtf.NAngMom)] 
    n = [0.0e0 for i in range(0,shell_pair1.Centre2.Cgtf.NAngMom)]
    ln = [copy.deepcopy(n) for i in range(0,shell_pair2.Centre1.Cgtf.NAngMom)]
    sln = [copy.deepcopy(ln) for i in range(0,shell_pair2.Centre2.Cgtf.NAngMom)]
    msln = [copy.deepcopy(sln) for i in range(0,shell_pair1.Centre1.Cgtf.NAngMom)] 
    contracted_coulomb = mnls
    contracted_exchange = msln
    # -------------------------------------------------------------------------------- #
    #   Loop over primitives in each CGTF, with no screening for now                   #
    # -------------------------------------------------------------------------------- #
    A = shell_pair1.Centre1
    B = shell_pair1.Centre2
    C = shell_pair2.Centre1
    D = shell_pair2.Centre2
    for primA in A.Cgtf.Primitives:
       for primB in B.Cgtf.Primitives:
          for primC in C.Cgtf.Primitives:
             for primD in D.Cgtf.Primitives:
                coulomb_shell_quartet = ShellQuartet(A,B,C,D,primA,primB,primC,primD)
                exchange_shell_quartet = ShellQuartet(A,D,C,B,primA,primD,primC,primB)
                evaluate_2e_integrals(coulomb_shell_quartet,contracted_coulomb)
                evaluate_2e_integrals(exchange_shell_quartet,contracted_exchange)
    return contracted_coulomb,contracted_exchange

# ================================================================================================ #
#  TWO-ELECTRON REPULSION INTEGRALS - MAIN SUBROUTINE                                              #
# ================================================================================================ #
def evaluate_2e_integrals(shell_quartet,contracted_coulomb):
 # -------------------------------------------------------------------------------- #
 #           Set up HRR & VRR indices and intermediates for bra and ket             #
 # -------------------------------------------------------------------------------- #
    bra_ket = ReIndex(shell_quartet)
    order = SetRR(bra_ket) 
 # -------------------------------------------------------------------------------- #
 #                        Initialize fundamental integrals                          #
 # -------------------------------------------------------------------------------- #
#    print 'bra_ket.ltot',bra_ket.ltot
#    print 'bra_ket ang mom',bra_ket.l1,bra_ket.l2,bra_ket.l3,bra_ket.l4
    coulomb = [{} for i in range(0,bra_ket.ltot+1)]
    lkey = (0,0,0,0,0,0,0,0,0,0,0,0) 
    Gab = shell_quartet.Gab; Gcd = shell_quartet.Gcd; T = shell_quartet.T
    zeta = shell_quartet.zeta; eta = shell_quartet.eta; tau = shell_quartet.tau
    F = F_aux(bra_ket.ltot,T)
#    print 'zeta,eta,tau,Gab,Gcd,T,rho,P,Q,W,rA,rB,rC,rD,lA,lB,lC,lD'
#    print shell_quartet.zeta,shell_quartet.eta,shell_quartet.tau,shell_quartet.Gab,shell_quartet.Gcd,\
#          shell_quartet.T,shell_quartet.rho,shell_quartet.Pab,shell_quartet.Pcd,\
#          shell_quartet.W,shell_quartet.rA,shell_quartet.rB,shell_quartet.rC,shell_quartet.rD,\
#          shell_quartet.lA,shell_quartet.lB,shell_quartet.lC,shell_quartet.lD
    for m in range(0,bra_ket.ltot+1):
       coulomb[m][lkey] = Gab*Gcd*2*pi**2.5e0*F[m]/(zeta*eta*sqrt(tau))
 # -------------------------------------------------------------------------------- #
 #          VRR for ket, on centre with highest angular momentum function           #
 # -------------------------------------------------------------------------------- #
    coulomb = vrr(coulomb,bra_ket,order.vrr_ket,'ket')
 # -------------------------------------------------------------------------------- #
 #          VRR for bra, on centre with highest angular momentum function           #
 # -------------------------------------------------------------------------------- #
    coulomb = vrr(coulomb,bra_ket,order.vrr_bra,'bra')
 # -------------------------------------------------------------------------------- #
 #                      HRR for ket and then HRR for bra                            #
 # -------------------------------------------------------------------------------- #
    coulomb = hrr(coulomb,bra_ket,order.hrr_ket,'ket')
    coulomb = hrr(coulomb,bra_ket,order.hrr_bra,'bra')
 # -------------------------------------------------------------------------------- #
 #                              Contraction                                         #
 # -------------------------------------------------------------------------------- #
    lA = shell_quartet.lA; lB = shell_quartet.lB
    lC = shell_quartet.lC; lD = shell_quartet.lD
    cA = shell_quartet.cA; cB = shell_quartet.cB
    cC = shell_quartet.cC; cD = shell_quartet.cD
    alpha = shell_quartet.alpha; beta = shell_quartet.beta
    gamma = shell_quartet.gamma; delta = shell_quartet.delta
    # print lA,lB,lC,lD,bra.ltot1,ket.ltot1
    # print 'two-electron integrals', coulomb
#    print 'contracted_coulomb',contracted_coulomb
    for (ilA,qA) in enumerate(c.lQuanta[lA]):
       for (ilB,qB) in enumerate(c.lQuanta[lB]):
          for (ilC,qC) in enumerate(c.lQuanta[lC]):
             for (ilD,qD) in enumerate(c.lQuanta[lD]):
                nA = normalize(alpha,qA)
                nB = normalize(beta,qB)
                nC = normalize(gamma,qC)
                nD = normalize(delta,qD)
                lkey = make_key([qA,qB,qC,qD])
#                if lA == lB == lC == lD == 1:
#                   print lkey,coulomb[0][lkey]
                cc = nA*nB*nC*nD*cA*cB*cC*cD
                contracted_coulomb[ilA][ilB][ilC][ilD] += cc*coulomb[0][lkey]
#                print ilA,ilB,ilC,ilD, lkey,coulomb[0][lkey], contracted_coulomb[ilA],contracted_coulomb[ilA][ilB][ilC][ilD]
#    print 'contracted_coulomb',contracted_coulomb
    return 

# ================================================================================================ #
#  TWO-ELECTRON REPULSION INTEGRALS - VRR AND HRR SUBROUTINES                                      #
# ================================================================================================ #
def vrr(coulomb,bk,order,mode):
#    print 'vrr'
    if mode == 'bra':
       i1 = bk.i1; i2 = bk.i2; i3 = bk.i3; i4 = bk.i4 # i1 & i3 are pointers to centres with highest angular momentum in bra and ket
       r = bk.r1; z = bk.z; P = bk.P
       index1 = 0; index3 = 2
    elif mode == 'ket':
       i1 = bk.i3; i2 = bk.i4; i3 = bk.i1; i4 = bk.i2
       r = bk.r3; z = bk.e; P = bk.Q
       index1 = 2; index3 = 0
    else:
       print 'Error: mode for VRR must be bra or ket'
    tau = bk.tau; W = bk.W; rho = bk.rho
    for [term,m] in order:
       l1 = term[index1]; l3 = term[index3] 
       lquartet = [[],[],[],[]]
       for qm in c.lQuanta[l1]:                       # qm are angular momentum quantum numbers for each Cartesian component of a given angular momentum, l1
          for ql in c.lQuanta[l3]:                    # ql are angular momentum quantum numbers for each Cartesian component of a given angular momentum, l3
             lquartet[i1] = qm[:]
             lquartet[i2] = c.lQuanta[0][0][:]        # q0 by definition for VRR 
             lquartet[i3] = ql[:]
             lquartet[i4] = c.lQuanta[0][0][:]        # q0 by definition for VRR
             lkey = make_key(lquartet)
             for j in range(0,3):                     # choose Cartesian component index to decrement 
                if qm[j] != 0:
                   j1 = j                             # j is Cartesian component index
             lquartet[i1][j1] -= 1                    # do decrementing in-place, sequentially
             ln1 = lquartet[i1][j1]
             ln3 = lquartet[i3][j1]
             lk0 = make_key(lquartet)
             lquartet[i1][j1] -= 1
             lk1 = make_key(lquartet)
             lquartet[i1][j1] += 1
             lquartet[i3][j1] -= 1
             lk3 = make_key(lquartet)
             coulomb[m][lkey] = (P[j1]-r[j1])*get(coulomb[m],lk0) \
                              + (W[j1]-P[j1])*get(coulomb[m+1],lk0)
             if ln1 > 0:
                coulomb[m][lkey] += ln1*(get(coulomb[m],lk1)-rho*get(coulomb[m+1],lk1)/z)/(2*z)
             if ln3 > 0:
                coulomb[m][lkey] += ln3*(get(coulomb[m+1],lk3))/(2*tau) \
             # print 'lkey vrr',lkey
             # print 'vrr', lkey, P, r, W, rho, tau, z, coulomb[m][lkey]
    return coulomb

def hrr(coulomb,bk,order,mode):
#    print 'hrr'
    if mode == 'bra':
       i1 = bk.i1; i2 = bk.i2; i3 = bk.i3; i4 = bk.i4
       r1 = bk.r1; r2 = bk.r2; r3 = bk.r3; r4 = bk.r4
       index1 = 0; index2 = 1; index3 = 2; index4 = 3 
    elif mode == 'ket':
       i1 = bk.i3; i2 = bk.i4; i3 = bk.i1; i4 = bk.i2
       r1 = bk.r3; r2 = bk.r4; r3 = bk.r1; r4 = bk.r2
       index1 = 2; index2 = 3; index3 = 0; index4 = 1 
    else:
       print 'Error: mode for HRR must be bra or ket'
    for term in order:
       l1 = term[index1]; l2 = term[index2]; l3 = term[index3]; l4 = term[index4]
       lquartet = [[],[],[],[]]
       for qm in c.lQuanta[l1]:
          for qn in c.lQuanta[l2]: 
             for ql in c.lQuanta[l3]:
                for qs in c.lQuanta[l4]:
                   lquartet[i1] = qm[:]
                   lquartet[i2] = qn[:]
                   lquartet[i3] = ql[:]
                   lquartet[i4] = qs[:] 
                   lkey = make_key(lquartet)
                   for j in range(0,3):
                      if qn[j] != 0:
                         j2 = j
                   lquartet[i2][j2] -= 1
                   lk0 = make_key(lquartet)
                   lquartet[i1][j2] += 1
                   lk1 = make_key(lquartet)
                   coulomb[0][lkey] = (r1[j2]-r2[j2])*get(coulomb[0],lk0) + get(coulomb[0],lk1)
                   # print 'lkey hrr',lkey
                   # print 'hrr', lkey, r1, r2, lk0, lk1, coulomb[0][lkey]
    return coulomb

# ================================================================================================ #
#  USEFUL SUBROUTINES
# ================================================================================================ #

def make_key(llists):
    lkey = tuple(itertools.chain.from_iterable(llists))
    return lkey

def distance(v1,v2):
    [x1,y1,z1] = v1
    [x2,y2,z2] = v2
    rsq = (x1-x2)**2+(y1-y2)**2+(z1-z2)**2
    return rsq

def F_aux(m,U):
    eps = 1.e-15
    F = [0.0 for i in range(0,m+1)]
    # compute F for maximum value of m then use downward recurrence relation
    # F_m[U] = (exp(-U)+2*U*F_(m+1)[U])/(2m+1) 
    mm = m+0.5
    mm2 = 2*mm
    if abs(U) < eps:
       f = 1/mm2
    elif abs(U) > 10.0:
       f = gamma(mm)/(2*U**mm)
    else:
       f = 0.5*(U**-mm)*gamma(mm)*gammainc(mm,U)
#       f = 0.0
#       for i in range(0,11):
#          f += (-U)**i/(factorial(i)*(mm2+2*i)) 
    F[m] = f
    for i in range(0,m):
       index = m-(i+1) 
       F[index] = (exp(-U)+2*U*F[index+1])/(2*index+1)
#       F[index] = (exp(-U)+2*U*F[index+1])/mm2 
    return F

def normalize(double exponent,lv):
    cdef double lx, ly, lz, coeffs 
    [lx,ly,lz] = lv
    coeff = (2.0e0*exponent)**((lx+ly+lz+1.5e0)/2.0e0)/ \
            sqrt(tgamma(lx+0.5e0)*tgamma(ly+0.5e0)*tgamma(lz+0.5e0))
    return coeff

def get(integral_dict,key):
    # get integral values from dictionary, initializing entries for keys containing negative recurrence indices as we go
    try:
       val = integral_dict[key]
    except:
       if -1 not in key:
          print 'ERROR: KEY NOT FOUND', key
          sys.exit()
       else:
#          print 'Initializing RR for key', key
          val = 0.0e0
    return val

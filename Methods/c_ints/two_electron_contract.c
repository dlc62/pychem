#include "two_electron_contract.h"

void two_electron_contract(double **contracted_ints, double **primitive_ints, 
                           double **cc_bra, double **cc_ket, 
                           int na, int nb, int nc, int nd, int lbra, int lket) {

   // Indices for primitives
   int ia, ib, ic, id; 
   // Angular momentum indices
   int ibra_x, ibra_y, ibra_z, iket_x, iket_y, iket_z; 
   // Integral indices
   int ibra, iket, ibra_c, iket_c;
 
   // Loop over all Cartesian basis functions, for all combinations of primitives

   ibra = -1; ibra_c = -1;

   // Start loop over bra
   for (ibra_x = lbra; ibra_x > -1; ibra_x--) {
     for (ibra_y = lbra-ibra_x; ibra_y > -1; ibra_y--) {
       ibra_z = lbra - ibra_x - ibra_y;

       ibra_c += 1;

       // Loop over all primitives for this angular momentum quantum state
       for (ia = 0; ia < na; ia++) {
         for (ib = 0; ib < nb; ib++) { 

           ibra += 1; iket = -1; iket_c = -1;

           // Start loop over ket
           for (iket_x = lket; iket_x > -1; iket_x--) {
             for (iket_y = lket-iket_x; iket_y > -1; iket_y--) {
               iket_z = lket - iket_x - iket_y;

               iket_c += 1;

               // Loop over all primitives for this angular momentum quantum state
               for (ic = 0; ic < nc; ic++) { 
                 for (id = 0; id < nd; id++) { 
                   
                   iket += 1;
  
                   // Do contraction, accumulate contracted integrals
                   contracted_ints[ibra_c][iket_c] += cc_bra[ia][ib]*cc_ket[ic][id]*primitive_ints[ibra][iket];

                 }  // end
               }    // loops
             }      // over
           }        // ket

         }  // end
       }    // loops
     }      // over
   }        // bra
 
} // end subroutine

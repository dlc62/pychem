#include "one_electron_contract.h"

void one_electron_contract(double **contracted_ints, double **primitive_ints, 
                           double **cc, int na, int nb, int la, int lb) {

   // Indices for primitives
   int ia, ib, ibra, iket; 
   // Angular momentum indices
   int iax, iay, iaz, ibx, iby, ibz; 
   // Contracted integral indices
   int ibra_c, iket_c;
 
   // Loop over all Cartesian basis functions, for all combinations of primitives

   ibra_c = -1;
   ibra = -1;

   // Start loop over bra, first over angular momentum quantum numbers then over primitives
   for (iax = la; iax > -1; iax--) {
     for (iay = la-iax; iay > -1; iay--) {
       iaz = la - iax - iay;

       ibra_c += 1;

       for (ia = 0; ia < na; ia++) {

         ibra += 1; iket = -1; iket_c = -1;

         // Start loop over ket, first over angular momentum quantum numbers then over primitives
         for (ibx = lb; ibx > -1; ibx--) {
           for (iby = lb-ibx; iby > -1; iby--) {
             ibz = lb - ibx - iby;

             iket_c += 1;

             for (ib = 0; ib < nb; ib++) { 

               iket += 1;

               // Do contraction, accumulate contracted integrals
               contracted_ints[ibra_c][iket_c] += cc[ia][ib]*primitive_ints[ibra][iket];

             }    // end loops
           }      // over
         }        // ket

       }    // end loops
     }      // over
   }        // bra
 
} // end subroutine

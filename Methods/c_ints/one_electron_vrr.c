#include "one_electron_vrr.h"
#include "angmom_index.h"

void one_electron_vrr(double **target_ints, double **base0, double **base1, 
                      double **base2, double **base3, double **sigma_P, double ***P, 
                      double *Ra, double *Rc, int na, int nb, int la, int aux) {

   // Indices for primitives
   int ia, ib; 
   // Recurrence relation indices
   int iax, iay, iaz, iax0, iay0, iaz0, iax1, iay1, iaz1;
   int index, index_a_val;
   // Integral indices
   int ibra, iket, ibra0, ibra1;
   // Logical
   int do_at1;
 
   // Loop over all Cartesian basis functions, for all combinations of primitives

   ibra = -1;

   // Start loop over bra
   for (iax = la; iax > -1; iax--) {
     for (iay = la-iax; iay > -1; iay--) {
       iaz = la - iax - iay;

       do_at1 = 0;
       // Set up indices for recurrence relations
       if (iax != 0) { 
          iax0 = iax-1; iay0 = iay; iaz0 = iaz; index = 0; index_a_val = iax0; 
          if (iax0 != 0) {
             iax1 = iax0-1; iay1 = iay0; iaz1 = iaz0; do_at1 = 1; 
          }
       } else if (iay != 0) { 
          iax0 = iax; iay0 = iay-1; iaz0 = iaz; index = 1; index_a_val = iay0; 
          if (iay0 != 0) {
             iax1 = iax0; iay1 = iay0-1; iaz1 = iaz0; do_at1 = 1;
          }
       } else if (iaz != 0) { 
          iax0 = iax; iay0 = iay; iaz0 = iaz-1; index = 2; index_a_val = iaz0;
          if (iaz0 != 0) {
             iax1 = iax0; iay1 = iay0; iaz1 = iaz0-1; do_at1 = 1;
          }
       } else {}
           
       // Loop over all primitives for this angular momentum quantum state
       for (ia = 0; ia < na; ia++) {

         ibra += 1; iket = -1;
         // Compute indices into base integral arrays
         ibra0 = angmom_index(iax0, iay0, iaz0)*na + ia;
         if (do_at1 == 1) { ibra1 = angmom_index(iax1, iay1, iaz1)*na + ia; } 

         // Loop over primitives in ket (l=0)
         for (ib = 0; ib < nb; ib++) { 
 
           iket += 1;

           if ( aux == 0 ) {
           // Compute overlap integrals
             target_ints[ibra][iket] = (P[ia][ib][index] - Ra[index])*base0[ibra0][ib]; 
             if (do_at1 == 1) {
               target_ints[ibra][iket] += index_a_val*0.5*sigma_P[ia][ib]*base1[ibra1][ib];
             } 

           } else {
           // Compute nuclear attraction integrals
             target_ints[ibra][iket] = (P[ia][ib][index] - Ra[index])*base0[ibra0][ib] 
                                      -(P[ia][ib][index] - Rc[index])*base1[ibra0][ib];
             if (do_at1 == 1) {
               target_ints[ibra][iket] += index_a_val*0.5*sigma_P[ia][ib]*(base2[ibra1][ib]-base3[ibra1][ib]);
             } 

           }

         }  // end loop over ket

       }    // end loops
     }      // over
   }        // bra
 
} // end subroutine

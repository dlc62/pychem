#include "one_electron_hrr.h"
#include "angmom_index.h"

void one_electron_hrr(double **target_ints, double **base0, double **base1, 
                      double *Rx, int na, int nb, int la, int lb) { 

   // Recurrence relation indices
   int iax, iay, iaz, ibx, iby, ibz; 
   int iax1, iay1, iaz1, ibx0, iby0, ibz0;
   int index, ila0, ila1, ilb0, ilb1;
   // Integral indices
   int ia0, ia1, ib0, ib1;
   // Primitive indices
   int ia, ib;
 
   // Loop over all contracted Cartesian basis functions and primitives within them
   // Start loop over ket, decrement b indices
   for (ibx = lb; ibx > -1; ibx--) {
     for (iby = lb-ibx; iby > -1; iby--) {
       ibz = lb - ibx - iby;

       // Set up indices for recurrence relations
       if (ibx != 0) { 
          ibx0 = ibx-1; iby0 = iby; ibz0 = ibz; index = 0; 
       } else if (iby != 0) { 
          ibx0 = ibx; iby0 = iby-1; ibz0 = ibz; index = 1; 
       } else if (ibz != 0) { 
          ibx0 = ibx; iby0 = iby; ibz0 = ibz-1; index = 2;
       } else { /* print error message, should not be able to get here */ }

       // Compute angular momentum indices (distinct from Cartesian component index)
       ilb0 = angmom_index(ibx0, iby0, ibz0);
       ilb1 = angmom_index(ibx, iby, ibz);
           
       // Loop over bra, increment corresponding a indices
       for (iax = la; iax > -1; iax--) {
         for (iay = la-iax; iay > -1; iay--) {
           iaz = la - iax - iay;

           // Set up indices for recurrence relations
           if (index == 0) { 
              iax1 = iax+1; iay1 = iay; iaz1 = iaz;
           } else if (index == 1) { 
              iax1 = iax; iay1 = iay+1; iaz1 = iaz;
           } else if (index == 2) { 
              iax1 = iax; iay1 = iay; iaz1 = iaz+1;
           } else { /* print error message, should not be able to get here */ }

           // Compute angular momentum indices
           ila0 = angmom_index(iax, iay, iaz);
           ila1 = angmom_index(iax1, iay1, iaz1);

           // Loop over primitives, compute indices into target (ia0,ib1), base0 (ia1,ib0) and base1 (ia0,ib0) arrays
           for (ia = 0; ia < na; ia++) { 

             ia0 = ila0*na + ia;
             ia1 = ila1*na + ia;

             for (ib = 0; ib < nb; ib++) {

               ib0 = ilb0*nb + ib;
               ib1 = ilb1*nb + ib;

               target_ints[ia0][ib1] = base0[ia1][ib0] + Rx[index]*base1[ia0][ib0]; 

             }  // end loops over 
           }    // primitives 

         }  // end loops 
       }    // over bra

     }   // end loops 
   }     // over ket
 
} // end subroutine

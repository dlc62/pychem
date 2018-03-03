#include "two_electron_hrr.h"
#include "angmom_index.h"

void two_electron_hrr(double **target_ints, double **base0, double **base1, 
                      double *Rx, int la, int lb, int lc, int ld, int goofy) { 

   // Recurrence relation indices
   int iax, iay, iaz, ibx, iby, ibz; 
   int icx, icy, icz, idx, idy, idz; 
   int index, iax_off, iay_off, iaz_off, ibx_off, iby_off, ibz_off;
   int ia0, ia1, ib0, ib1;
   // Integral indices
   int ibra, ibra0, ibra1, iket;
   int nlb, nlb0, nlb1;
 
   nlb = angmom_index(0,0,lb)+1;
   if (goofy == 0) { lb += 1; nlb0 = nlb; nlb1 = nlb; }
   if (goofy == 1) { la += 1; nlb0 = angmom_index(0,0,lb+1); nlb1 = nlb; }

   ibra = -1; 

   // Loop over all contracted Cartesian basis functions
   // incrementing and decrementing angular momentum indices as required

   for (iax = la; iax > -1; iax--) {
     for (iay = la-iax; iay > -1; iay--) {
       iaz = la - iax - iay;

       if (goofy == 1) {
         // Set up indices for recurrence relations (goofy)
         iax_off = 0; iay_off = 0; iaz_off = 0;
         ibx_off = 0; iby_off = 0; ibz_off = 0;
         if (iax != 0) { 
           iax_off = -1; ibx_off = +1; index = 0; 
         } else if (iay != 0) { 
           iay_off = -1; iby_off = +1; index = 1;
         } else if (iaz != 0) { 
           iaz_off = -1; ibz_off = +1; index = 2;
         } else { /* do nothing */ }
       }    

       for (ibx = lb; ibx > -1; ibx--) {
         for (iby = lb-ibx; iby > -1; iby--) {
           ibz = lb - ibx - iby;

           // Set up indices for recurrence relations (regular)
           if (goofy == 0) { 
             iax_off = 0; iay_off = 0; iaz_off = 0;
             ibx_off = 0; iby_off = 0; ibz_off = 0;
             if (ibx != 0) { 
               ibx_off = -1; iax_off = +1; index = 0; 
             } else if (iby != 0) { 
               iby_off = -1; iay_off = +1; index = 1;
             } else if (ibz != 0) { 
               ibz_off = -1; iaz_off = +1; index = 2;
             } else { /* do nothing */ }
           }

           // Compute indices into base integral arrays
           ia0 = angmom_index(iax+iax_off,iay+iay_off,iaz+iaz_off);
           ib0 = angmom_index(ibx+ibx_off,iby+iby_off,ibz+ibz_off);
           if (goofy == 0) { ib1 = ib0; ia1 = angmom_index(iax,iay,iaz); }
           if (goofy == 1) { ia1 = ia0; ib1 = angmom_index(ibx,iby,ibz); }
           ibra0 = ia0*nlb0 + ib0;
           ibra1 = ia1*nlb1 + ib1;

           ibra += 1; iket = -1;

//           printf("ibra indices: %i %i %i %i\n", ibra, ibra0, ibra1, index);

           // Loop over ket, simply accumulating the iket index
           for (icx = lc; icx > -1; icx--) {
             for (icy = lc-icx; icy > -1; icy--) {
               icz = lc - icx - icy;

               for (idx = ld; idx > -1; idx--) {
                 for (idy = ld-idx; idy > -1; idy--) {
                   idz = ld - idx - idy;

                   iket += 1;

                   target_ints[ibra][iket] = base0[ibra0][iket] + Rx[index]*base1[ibra1][iket]; 

                 }  // end
               }    // loops
             }      // over
           }        // ket

         }  // end
       }    // loops
     }      // over
   }        // bra
 
} // end subroutine

#include "one_electron_kinetic.h"
#include "angmom_index.h"

void one_electron_kinetic(double **target_ints, double **base0, double **base1, 
                          double **base2, double *exB, int na, int nb, int la, int lb) { 

   // Recurrence relation indices
   int iax, iay, iaz, ibx, iby, ibz; 
   int ilbx_p2, ilby_p2, ilbz_p2, ilbx_m2, ilby_m2, ilbz_m2, ila, ilb; 
   // Integral indices
   int ibra, iket, ibx_p2, iby_p2, ibz_p2, ibx_m2, iby_m2, ibz_m2;
   // Primitive indices
   int ia, ib;
   // Intermediates
   double beta;
 
   // Loop over all contracted Cartesian basis functions and primitives within them
   // Start loop over angular momentum class cgtfs in ket
   for (ibx = lb; ibx > -1; ibx--) {
     for (iby = lb-ibx; iby > -1; iby--) {
       ibz = lb - ibx - iby;

       // Compute angular momentum indices (distinct from Cartesian component index)
       ilbx_p2 = angmom_index(ibx+2, iby, ibz);
       ilby_p2 = angmom_index(ibx, iby+2, ibz);
       ilbz_p2 = angmom_index(ibx, iby, ibz+2);

       ilb = angmom_index(ibx, iby, ibz);

       ilbx_m2 = -1; ilby_m2 = -1; ilbz_m2 = -1;
       if (ibx > 1) { ilbx_m2 = angmom_index(ibx-2, iby, ibz); }
       if (iby > 1) { ilby_m2 = angmom_index(ibx, iby-2, ibz); }
       if (ibz > 1) { ilbz_m2 = angmom_index(ibx, iby, ibz-2); }

       // Loop over angular momentum class cgtfs in bra
       for (iax = la; iax > -1; iax--) {
         for (iay = la-iax; iay > -1; iay--) {
           iaz = la - iax - iay;

           // Compute angular momentum index
           ila = angmom_index(iax, iay, iaz);

           ibx_m2 = -1; iby_m2 = -1; ibz_m2 = -1;
           // Loop over primitives, compute indices into target and base arrays
           for (ib = 0; ib < nb; ib++) {

             beta = exB[ib];
             iket = ilb*nb + ib;
             ibx_p2 = ilbx_p2*nb + ib;
             iby_p2 = ilby_p2*nb + ib;
             ibz_p2 = ilbz_p2*nb + ib;
             if (ilbx_m2 > -1) { ibx_m2 = ilbx_m2*nb + ib; }
             if (ilby_m2 > -1) { iby_m2 = ilby_m2*nb + ib; }
             if (ilbz_m2 > -1) { ibz_m2 = ilbz_m2*nb + ib; }

             for (ia = 0; ia < na; ia++) { 

               ibra = ila*na + ia;

               target_ints[ibra][iket] = beta*(2*ibx+2*iby+2*ibz+3)*base1[ibra][iket];
               target_ints[ibra][iket] -= 2*beta*beta*(base0[ibra][ibx_p2]+base0[ibra][iby_p2]+base0[ibra][ibz_p2]);
               if (ilbx_m2 > -1) { target_ints[ibra][iket] -= 0.5*ibx*(ibx-1)*base2[ibra][ibx_m2]; } 
               if (ilby_m2 > -1) { target_ints[ibra][iket] -= 0.5*iby*(iby-1)*base2[ibra][iby_m2]; } 
               if (ilbz_m2 > -1) { target_ints[ibra][iket] -= 0.5*ibz*(ibz-1)*base2[ibra][ibz_m2]; } 
  
             }  // end loops over 
           }    // primitives 

         }  // end loops 
       }    // over bra

     }   // end loops 
   }     // over ket
 
} // end subroutine

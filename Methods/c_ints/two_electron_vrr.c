#include "two_electron_vrr.h"
#include "angmom_index.h"

void two_electron_vrr(double **target_ints, double **base0, double **base1, 
                      double **base2, double **base3, double **base4, 
                      double **zeta, double **eta, double *kappa, double *Rx, double ***R, 
                      int na, int nb, int nc, int nd, int lbra, int lket, int kappa_index) {
   // Intermediates
   double base, a_inc, c_inc;
   // Indices for primitives
   int ia, ib, ic, id; 
   // Recurrence relation indices
   int ibra_x, ibra_y, ibra_z, iket_x, iket_y, iket_z; 
   int ibra_x0, ibra_y0, ibra_z0, ibra_x1, ibra_y1, ibra_z1, iket_x1, iket_y1, iket_z1;
   int index, index_a_val, index_c_val;
   // Integral indices
   int ibra, iket, ibra0, ibra1, iket1;
   int ibra_off, ibra0_off, ibra1_off, iket_off, iket1_off;
   // Primitive counters
   int ibra_prim, iket_prim;
   // Logical
   int do_at1, do_ct1;
 
   // Loop over all Cartesian basis functions, for all combinations of primitives

   // Start loop over bra
   for (ibra_x = lbra; ibra_x > -1; ibra_x--) {
     for (ibra_y = lbra-ibra_x; ibra_y > -1; ibra_y--) {
       ibra_z = lbra - ibra_x - ibra_y;

       do_at1 = 0;
       // Set up indices for recurrence relations
       if (ibra_x != 0) { 
          ibra_x0 = ibra_x-1; ibra_y0 = ibra_y; ibra_z0 = ibra_z; index = 0; index_a_val = ibra_x0; 
          if (ibra_x0 != 0) {
             ibra_x1 = ibra_x0-1; ibra_y1 = ibra_y0; ibra_z1 = ibra_z0; do_at1 = 1; 
          }
       } else if (ibra_y != 0) { 
          ibra_x0 = ibra_x; ibra_y0 = ibra_y-1; ibra_z0 = ibra_z; index = 1; index_a_val = ibra_y0; 
          if (ibra_y0 != 0) {
             ibra_x1 = ibra_x0; ibra_y1 = ibra_y0-1; ibra_z1 = ibra_z0; do_at1 = 1;
          }
       } else if (ibra_z != 0) { 
          ibra_x0 = ibra_x; ibra_y0 = ibra_y; ibra_z0 = ibra_z-1; index = 2; index_a_val = ibra_z0;
          if (ibra_z0 != 0) {
             ibra_x1 = ibra_x0; ibra_y1 = ibra_y0; ibra_z1 = ibra_z0-1; do_at1 = 1;
          }
       } else {}
           
       ibra_off = angmom_index(ibra_x, ibra_y, ibra_z)*na*nb;
       ibra0_off = angmom_index(ibra_x0, ibra_y0, ibra_z0)*na*nb;
       if (do_at1 == 1) { ibra1_off = angmom_index(ibra_x1, ibra_y1, ibra_z1)*na*nb; }

       // Loop over all primitives for this angular momentum quantum state
       ibra_prim = -1;
       for (ia = 0; ia < na; ia++) {
         for (ib = 0; ib < nb; ib++) { 

           ibra_prim += 1;
           ibra = ibra_off + ibra_prim;
           ibra0 = ibra0_off + ibra_prim;
           if (do_at1 == 1) { ibra1 = ibra1_off + ibra_prim; }

           // Start loop over ket
           for (iket_x = lket; iket_x > -1; iket_x--) {
             for (iket_y = lket-iket_x; iket_y > -1; iket_y--) {
               iket_z = lket - iket_x - iket_y;

               // Set up indices for recurrence relations
               do_ct1 = 0;
               if (iket_x != 0 && index == 0) { 
                  iket_x1 = iket_x-1; iket_y1 = iket_y; iket_z1 = iket_z; index_c_val = iket_x; do_ct1 = 1;
               } else if (iket_y != 0 && index == 1) { 
                  iket_x1 = iket_x; iket_y1 = iket_y-1; iket_z1 = iket_z; index_c_val = iket_y; do_ct1 = 1;
               } else if (iket_z != 0 && index == 2) { 
                  iket_x1 = iket_x; iket_y1 = iket_y; iket_z1 = iket_z-1; index_c_val = iket_z; do_ct1 = 1;
               } else {}

               iket_off = angmom_index(iket_x, iket_y, iket_z)*nc*nd;
               if (do_ct1 == 1) { iket1_off = angmom_index(iket_x1, iket_y1, iket_z1)*nc*nd; } 

               iket_prim = -1; 
               // Loop over all primitives for this angular momentum quantum state
               for (ic = 0; ic < nc; ic++) { 
                 for (id = 0; id < nd; id++) { 
                   
                   iket_prim += 1;
                   iket = iket_off + iket_prim;
                   if (do_ct1 == 1) { iket1 = iket1_off + iket_prim; } 

                   // Compute index into base0 and base1 arrays (ibra0)
                   if (kappa_index == 0) { base = Rx[index]*kappa[ia]*zeta[ia][ib]*base0[ibra0][iket] + R[ibra_prim][iket_prim][index]*zeta[ia][ib]*base1[ibra0][iket]; }
                                    else { base = Rx[index]*kappa[ib]*zeta[ia][ib]*base0[ibra0][iket] + R[ibra_prim][iket_prim][index]*zeta[ia][ib]*base1[ibra0][iket]; }
//                   base = Rx[index]*kappa[ib]*zeta[ia][ib]*base0[ibra0][iket] + R[ibra_prim][iket_prim][index]*zeta[ia][ib]*base1[ibra0][iket]; 
                   target_ints[ibra][iket] = base;
//                   printf("base %16.12f\n", base);
//                   printf("base %16.12f %16.12f\n", Rx[index]*kappa[ib]*zeta[ia][ib]*base0[ibra0][iket] , R[ibra_prim][iket_prim][index]*zeta[ia][ib]*base1[ibra0][iket]); 
//                   printf("base %i %i %i %i %16.12f %16.12f %16.12f %16.12f \n", ia, ib, ic, id, base0[ibra0][iket], Rx[index]*kappa[ib]*zeta[ia][ib], Rx[index]*kappa[ia]*zeta[ia][ib], Rx[index]*kappa[ia]*zeta[ia][ib]*base0[ibra0][iket]); 

                   if (do_at1 == 1) {
                     // Compute index into base2 and base3 arrays (ibra1)
                     a_inc = index_a_val*zeta[ia][ib]*(base2[ibra1][iket]-zeta[ia][ib]*base3[ibra1][iket]); 
                     target_ints[ibra][iket] += a_inc;
                   }

                   if (do_ct1 == 1) {
                     // Compute index into base4 array (ibra0 with iket1)
                     c_inc = index_c_val*zeta[ia][ib]*eta[ic][id]*base4[ibra0][iket1]; 
                     target_ints[ibra][iket] += c_inc; 
//                     printf("ang mom indices, primitive indices, integral components, integral: %i %i %i %i %i %i, %i %i %i %i, %16.12f %16.12f %16.12f %16.12f %16.12f, %16.12f\n", 
//                           ibra_x, ibra_y, ibra_z, iket_x, iket_y, iket_z, ia, ib, ic, id, 
//                           Rx[index]*kappa[ia]*zeta[ia][ib]*base0[ibra0][iket], R[ibra_prim][iket_prim][index]*zeta[ia][ib]*base1[ibra0][iket],
//                           index_a_val*zeta[ia][ib]*(base2[ibra1][iket]), -index_a_val*zeta[ia][ib]*zeta[ia][ib]*base3[ibra1][iket],
//                           index_c_val*zeta[ia][ib]*eta[ic][id]*base4[ibra0][iket1], target_ints[ibra][iket]); 
//                     printf("c_inc %16.12f\n", c_inc);
//                   } else {
//                     if (do_at1 == 1) {
//                       printf("ang mom indices, primitive indices, integral components, integral: %i %i %i %i %i %i, %i %i %i %i, %16.12f %16.12f %16.12f %16.12f, %16.12f\n", 
//                             ibra_x, ibra_y, ibra_z, iket_x, iket_y, iket_z, ia, ib, ic, id, 
//                             Rx[index]*kappa[ia]*zeta[ia][ib]*base0[ibra0][iket], R[ibra_prim][iket_prim][index]*zeta[ia][ib]*base1[ibra0][iket],
//                             index_a_val*zeta[ia][ib]*(base2[ibra1][iket]), -index_a_val*zeta[ia][ib]*zeta[ia][ib]*base3[ibra1][iket], target_ints[ibra][iket]); 
//                     } else {
//                       printf("ang mom indices, primitive indices, integral components, integral: %i %i %i %i %i %i, %i %i %i %i, %16.12f %16.12f %16.12f \n", 
//                             ibra_x, ibra_y, ibra_z, iket_x, iket_y, iket_z, ia, ib, ic, id, 
//                             Rx[index]*kappa[ia]*zeta[ia][ib]*base0[ibra0][iket], R[ibra_prim][iket_prim][index]*zeta[ia][ib]*base1[ibra0][iket], target_ints[ibra][iket]); 
//                     }
                   }

//                   printf("integral %16.12f\n", target_ints[ibra][iket]);

                 }  // end
               }    // loops
             }      // over
           }        // ket

         }  // end
       }    // loops
     }      // over
   }        // bra
 
} // end subroutine

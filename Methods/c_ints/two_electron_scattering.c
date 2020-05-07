#include "two_electron_scattering.h"
#include "spherical_bessel_j.h"
#define _USE_MATH_DEFINES
#include <math.h>

void two_electron_scattering(double ***fundamentals, double **sigma_P, double **U_P, double ***P,
                             double **sigma_Q, double **U_Q, double ***Q, double ***R_array, double S, 
                             int na, int nb, int nc, int nd, int l_max) {

//-----------------------------------------------------------------------------

   int ia, ib, ic, id, i, ibra, iket, m; 
   double R, R2, theta_sq_inv, z, U, Ssq, Spow, df, eS, quarter_Ssq;
   double epsilon = 1.e-14;
   double f[29],S2[29],j[29];

   // Pre-compute intermediates that only depend on S

   Ssq = S*S; df = 1.0; Spow = 1.0; quarter_Ssq = 0.25*Ssq;
   f[0] = 1.0; S2[0] = 1.0;
   for (m = 1; m < l_max+1; m ++) {
      Spow *= Ssq;
      df *= (double)(2*m+1);
      S2[m] = Spow;
      f[m] = 1.0/df;
   }        

   // Loop over all combinations of primitives

   ibra = -1;

   for (ia = 0; ia < na; ia++) {
     for (ib = 0; ib < nb; ib++) { 

       ibra += 1; iket = -1;

       for (ic = 0; ic < nc; ic++) { 
         for (id = 0; id < nd; id++) { 

           iket += 1;

           U = U_P[ia][ib]*U_Q[ic][id];
           theta_sq_inv = (sigma_P[ia][ib]+sigma_Q[ic][id]);
           eS = exp(-quarter_Ssq*theta_sq_inv);

           R2 = 0;
           for (i = 0; i < 3; i++) {
             R = (P[ia][ib][i] - Q[ic][id][i]);
             R_array[ibra][iket][i] = R;
             R2 += R*R;
           } 

           if (S < epsilon) { // special case S = 0

             fundamentals[0][ibra][iket] = U;
             for (m = 1; m < l_max+1; m ++) {
               fundamentals[m][ibra][iket] = 0;
             }

           } else if (R2 < epsilon) { // special case R = 0, S != 0

             for (m = 0; m < l_max+1; m ++) {
               fundamentals[m][ibra][iket] = U*eS*S2[m]*f[m];
             }

           } else { // general z case

             z = S*sqrt(R2);
           
             // compute z^-m j_m(z) for 0 < m < l_max)
             spherical_bessel_j(j,z,l_max);

             for (m = l_max; m > -1; m --) {
               fundamentals[m][ibra][iket] = U*eS*S2[m]*j[m];
             }
 
           } // endif R cases

         }
       }
     }
   }

} // end subroutine

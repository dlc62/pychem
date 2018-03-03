#include "two_electron_fundamentals.h"
#include "interpolation_table.h"
#define _USE_MATH_DEFINES
#include <math.h>

void two_electron_fundamentals(double ***fundamentals, double **sigma_P, double **U_P, double ***P,
                               double **sigma_Q, double **U_Q, double ***Q, double ***R_array, 
                               int na, int nb, int nc, int nd, int l_max) {

// Implement interpolation table algorithm for calculating F_m(T)
// Note that m index comes first in fundamentals array
//-----------------------------------------------------------------------------
// The following interpolation table quantities are defined in
// interpolation_table.c and included through interpolation_table.h
// d = interpolation table step size
// n_points = number of points in interpolation tables
// f0, f1, f2, f3 = F_m(T)-derivative derived weights for use in interpolation
//-----------------------------------------------------------------------------

   int ia, ib, ic, id, i, j, ibra, iket, m; 
   double R, R2, theta_sq, two_theta_sq, T, sT, f, U;
   double epsilon = 1.e-14;
   double dd = 2*d;
   double pf = pow(2/M_PI,0.5); 

   // Loop over all combinations of primitives

   ibra = -1;

   for (ia = 0; ia < na; ia++) {
     for (ib = 0; ib < nb; ib++) { 

       ibra += 1; iket = -1;

       for (ic = 0; ic < nc; ic++) { 
         for (id = 0; id < nd; id++) { 

           iket += 1;

           U = U_P[ia][ib]*U_Q[ic][id];
           theta_sq = 1/(sigma_P[ia][ib]+sigma_Q[ic][id]);
           two_theta_sq = 2*theta_sq;

           R2 = 0;
           for (i = 0; i < 3; i++) {
             R = (P[ia][ib][i] - Q[ic][id][i]);
             R_array[ibra][iket][i] = R;
             R2 += R*R;
           } 

           if (R2 < epsilon) {

             // special case T = 0
             for (m = 0; m < l_max+1; m ++) {
               f = 1/(2*(double)m+1); 
               fundamentals[m][ibra][iket] = pf*U*pow(two_theta_sq,m+0.5)*f;
             }
//             printf("Small T: %16.12f %16.12f %16.12f %16.12f\n", pf*U*pow(two_theta_sq,m+0.5),f,0.0,fundamentals[0][ibra][iket]);

           } else {

             T = theta_sq*R2;
             sT = T/dd;
             j = (int) sT;
           
             if (j < n_points) {

               // general case interpolation
               for (m = l_max; m > -1; m --) {
                 f = f0[m][j] + sT*(f1[m][j] + sT*(f2[m][j] + sT*f3[m][j]));
                 fundamentals[m][ibra][iket] = pf*U*pow(two_theta_sq,m+0.5)*f;
                 if (m == 0) { 
//                    printf("General T: %16.12f %16.12f %16.12f %16.12f %16.12f\n", pf, U, pow(two_theta_sq,0.5), f, fundamentals[0][ibra][iket]);
                 }
               }
 
             } else {
            
               // asymptotic formula (large T)
               for (m = 0; m < l_max+1; m ++) {
                 f = tgamma(m+0.5)/(2*pow(T,m+0.5));
                 fundamentals[m][ibra][iket] = pf*U*pow(two_theta_sq,m+0.5)*f;
               }
//               printf("Large T: %16.12f %16.12f %16.12f %16.12f\n", pf*U*pow(two_theta_sq,m+0.5),f,T,fundamentals[0][ibra][iket]);
 
             } // endif - general case + asymptotic case (large T) 

           } // endif - special case (T = 0)

         }
       }
     }
   }

} // end subroutine

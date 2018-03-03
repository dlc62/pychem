#include "one_electron_fundamentals.h"
#include "interpolation_table.h"
#define _USE_MATH_DEFINES
#include <math.h>

void one_electron_fundamentals(double ***fundamentals, double **sigma_P, double **U_P, double ***P,
                               double *R_c, double Z, int na, int nb, int l_max) {

// Implement interpolation table algorithm for calculating F_m(T)
// Note that m index comes first in fundamentals array
//-----------------------------------------------------------------------------
// The following interpolation table quantities are defined in
// interpolation_table.c and included through interpolation_table.h
// d = interpolation table step size
// n_points = number of points in interpolation tables
// f0, f1, f2, f3 = F_m(T)-derivative derived weights for use in interpolation
// e0, e1, e2, e3 = Exp(-T)-derivative derived weights for use in interpolation
//-----------------------------------------------------------------------------

   // test this against python code

   int ia, ib, i, j, m; 
   double R, R2, T, sT, U, zeta, two_zeta, f, spf;
//   double R, R2, T, sT, U, zeta, two_zeta, f, f_prev, e, spf;
   double epsilon = 1.e-14;
   double dd = 2*d;
   double pf = pow(2/M_PI,0.5); 

   // Loop over all combinations of primitives

   pf = -Z*pf;

   for (ia = 0; ia < na; ia++) {
     for (ib = 0; ib < nb; ib++) { 

       R2 = 0;
       for (i = 0; i < 3; i++) {
         R = (P[ia][ib][i] - R_c[i]);
         R2 += R*R;
       } 

       U = U_P[ia][ib];
       zeta = 1.0/sigma_P[ia][ib];
       two_zeta = 2.0*zeta;
       spf = pf*U*pow(two_zeta,0.5);

       if (R2 < epsilon) {

         // special case T = 0
         for (m = 0; m < l_max+1; m ++) {
           f = 1/(2*(double)m+1); 
           fundamentals[m][ia][ib] = spf*f;
         }
//         printf("(small R) ia, ib, T, pf, f, int: %i, %i, %16.12f %16.12f %16.12f %16.12f \n", 
//                 ia, ib, 0.0, pf*pow(two_zeta,0.5), f, fundamentals[l_max][ia][ib]);

       } else {

         T = zeta*R2;
         sT = T/dd;
         j = (int) sT;
           
         if (j < n_points) {

           // general case interpolation
           for (m = 0; m < l_max+1; m++) {
              f = f0[m][j] + sT*(f1[m][j] + sT*(f2[m][j] + sT*f3[m][j]));
              fundamentals[m][ia][ib] = spf*f;
           }
//           f = f0[l_max][j] + sT*(f1[l_max][j] + sT*(f2[l_max][j] + sT*f3[l_max][j]));
//           fundamentals[l_max][ia][ib] = spf*f;
//           f_prev = f;
//           printf("(general) ia, ib, T, pf, f, int: %i, %i, %16.12f %16.12f %16.12f %16.12f \n", 
//                   ia, ib, T, pf*pow(two_zeta,0.5), f, fundamentals[l_max][ia][ib]);
 
//           if (l_max != 0) {
             // downward recursion for remaining m values
//             e = e0[j] + sT*(e1[j] + sT*(e2[j] + sT*e3[j]));
//             for (m = l_max - 1; m > -1; m--) {
//               f = e + 2*T*f_prev;
//               fundamentals[m][ia][ib] = pf*U*pow(two_zeta,m+0.5)*f;
//               f_prev = f;
//             }
//           }

         } else {
            
           // asymptotic formula (large T)
           for (m = 0; m < l_max+1; m ++) {
             f = tgamma(m+0.5)/(2*pow(T,m+0.5));
             fundamentals[m][ia][ib] = spf*f;
           }
//           printf("(large R) ia, ib, T, pf, f, int: %i, %i, %16.12f %16.12f %16.12f %16.12f \n", 
//                   ia, ib, T, pf*pow(two_zeta,0.5), f, fundamentals[l_max][ia][ib]);
 
         } // endif - general case + asymptotic case (large T) 

       } // endif - special case (T = 0)

     }
   }

} // end subroutine

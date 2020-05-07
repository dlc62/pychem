#include "shellpair_quantities.h"
#define _USE_MATH_DEFINES
#include <math.h>

void shellpair_quantities(double **sigmas, double **overlaps, double ***centres,
                          double *alpha_exponents, double *A, int n_alpha, 
                          double *beta_exponents, double *B, int n_beta) {

   int ia, ib, i; 
   double pi = M_PI;
   double r, r2, alpha, beta, sigma;

   // exponent-independent distance-squared between centres

   r2 = 0;
   for (i = 0; i < 3; i++) { 
      r = A[i]-B[i];
      r2 += r*r;
   }

   // exponent-dependent quantities

   for (ia = 0; ia < n_alpha; ia++) {
      for (ib = 0; ib < n_beta; ib++) { 

         alpha = alpha_exponents[ia];
         beta = beta_exponents[ib];
         sigma = 1.0/(alpha+beta);
         sigmas[ia][ib] = sigma;
         overlaps[ia][ib] = pow(pi*sigma,1.5)*exp(-alpha*beta*sigma*r2);
         for (i = 0; i < 3; i++) {
            centres[ia][ib][i] = (alpha*A[i]+beta*B[i])*sigma;
         } 

      } // end loop b over orbital exponents
   } // end loop a over orbital exponents

} // end subroutine

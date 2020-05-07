#include "two_electron_bound.h"
#define _USE_MATH_DEFINES
#include <math.h>

void two_electron_bound(double *bounds, double **C_P, double **C_Q, int nla, int nlb, int nlc, int nld) {

   int ia, ib, ic, id, n1, n2, n3, index; 

   n1 = nld;
   n2 = nlc*nld;
   n3 = nlb*nlc*nld;
   index = -1;

   for (ia = 0; ia < nla; ia++) {
     for (ib = 0; ib < nlb; ib++) { 
       for (ic = 0; ic < nlc; ic++) { 
         for (id = 0; id < nld; id++) { 

           index += 1;
           bounds[index] = sqrt(C_P[ia][ib]*C_Q[ic][id]); 

         }
       }
     }
   }

} // end subroutine

#include "spherical_bessel_j.h"
#define _USE_MATH_DEFINES
#include <math.h>

void spherical_bessel_j(double* j, double z, int l_max) {

   int m, k, k_max;
   int l_thresh = 16;
   double j0, jsum, df, mm1, mm, mm_df, zoff, zi, zinv, zinv2, zz, z2, sign, denom;
   double low = 1.e-3, med_low = 1.e-1, medium = 1.e0, med_high = 1.e1, high = 1.e2;
   double half_pi = 0.5*M_PI;
   double tiny = 1.e-16;

   if (z < med_high) { // Use series expansion about z = 0
      
      if (z < low) { k_max = 1; } 
      else if (z < med_low) { k_max = 3; }
      else if (z < medium) { k_max = 6; }
      else { k_max = 20; }

      z2 = z*z; df = 1.0;

      for (m = 0; m < l_max+1; m++) {

        mm1 = (double)(2*m+1); df *= mm1;

        jsum = 1.0; mm_df = 1.0; mm = mm1; zz = 1.0; sign = 1.0; denom = 1.0;

        for (k = 1; k < k_max+1; k++) {
          mm += 2; 
          mm_df *= mm;
          zz *= z2;
          sign *= -1;
          denom *= 2*k;
          jsum += sign*zz/(denom*mm_df); 
        }

        j[m] = jsum/df;
      }

   } else if (z > high) { // Use asymptotic expansion
  
      zinv = 1/z; 
      j[0] = sin(z)*zinv;

      zinv2 = zinv*zinv; zi = 1.0; zoff = z;
      for (m = 1; m < l_max+1; m++) {
        mm1 = (double)(m*(m+1)/2);
        zi *= zinv; 
        zoff -= half_pi; 
        j[m] = zi*(zinv*sin(zoff) + mm1*zinv2*cos(zoff));
      }

   } else { // Use upward recursion if l_max < l_thresh or downward recursion otherwise

      zinv = 1/z; zinv2 = zinv*zinv;
      j0 = sin(z)*zinv;
      j[0] = j0;
      j[1] = (sin(z) - z*cos(z))*zinv2;

      if (l_max < l_thresh) {
         
         for (m = 2; m < l_max + 1; m++) {
            j[m] = (2*m-1)*j[m-1]/z - j[m-2]; 
         }

      } else { 

         zoff = z-l_max*half_pi; mm1 = (double)(l_max*(l_max+1)/2);
         j[l_max] = pow(z,-l_max)*(zinv*sin(zoff)+mm1*zinv2*cos(zoff));

         zoff = z-(l_max-1)*half_pi; mm1 = (double)(l_max*(l_max-1)/2);
         j[l_max-1] = pow(z,-l_max+1)*(zinv*sin(zoff)+mm1*zinv2*cos(zoff));

         for (m = l_max; m > -1; m--) {
            j[m] = (2*m+3)*j[m+1] - j[m+2];
         }

         // is renormalization necessary if we start from reliable values of j[l_max] and j[l_max-1]?
         // check by printing j[0] and j0
      } 

   } // end cases - series expansion at low z, asymptotic expansion at high z and direct evaluation + recurrence relations otherwise 

   for (m = 0; m < l_max+1; m++) {
      if (fabs(j[m]) < tiny) { j[m] = 0.0; }
   }

}

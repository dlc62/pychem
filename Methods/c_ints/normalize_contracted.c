#include "normalize_contracted.h"
#include <math.h>

double normalize_contracted(double* nm, double* ex, double* cc, int ltot, int n) {
  
  double coeff, ltot, power, overlap;
  double glx, gly, glz;
  double lx, ly, lz;
  int index, i, j;

  power = (ltot+1.5);
  index = -1; 

  for (ilx = ltot; ilx > -1; ilx--) { 
    for (ily = ltot-ilx; ily > -1; ily--) {
      ilz = ltot - ilx - ily;

      index += 1;

      lx = (double) ilx; 
      ly = (double) ily; 
      lz = (double) ilz; 
      glx = tgamma(lx+0.5);
      gly = tgamma(ly+0.5);
      glz = tgamma(lz+0.5);

      overlap = 0.0

      for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
          overlap += cc[i]*cc[j]*pow((ex[i]+ex[j]),-power); 
        }
      }

      coeff = pow(glx*gly*glz*overlap,-0.5);

      nm[index] = coeff;

    }
  } 

  return nm;

} 

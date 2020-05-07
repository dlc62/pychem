#include "normalize.h"
#include <math.h>

double normalize(double exponent, int ilx, int ily, int ilz) {
  
  double coeff, ltot, power;
  double glx, gly, glz;
  double lx = (double) ilx; 
  double ly = (double) ily; 
  double lz = (double) ilz; 

  ltot = lx+ly+lz;
  power = (ltot+1.5)/2.0;
  glx = tgamma(lx+0.5);
  gly = tgamma(ly+0.5);
  glz = tgamma(lz+0.5);
 
  coeff = pow(2.0*exponent,power)/sqrt(glx*gly*glz);

  return coeff;

} 

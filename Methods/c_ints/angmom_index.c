#include "angmom_index.h"

int angmom_index(int lx, int ly, int lz) {
  
  int i, index;
 
  index = 0;
  for (i = 0; i < ly+lz+1; i++) {
    index += i;
  }
  index += lz;

  return index;

} 

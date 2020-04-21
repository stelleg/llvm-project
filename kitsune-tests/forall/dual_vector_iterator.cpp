#include <cstdio>
#include <vector>
#include <kitsune.h>

int main() {
  
  std::vector<int> vvv{23,24,25,26,27,28,29,30,31,32,33,34,35};
  std::vector<int> www{123,124,125,126,127,128,129,130,131,132,133,134,135};

  std::vector<int>::iterator vvv0=vvv.begin(), www0=www.begin();

  forall (std::vector<int>::iterator vvvi=vvv0, wwwi=www0; vvvi != vvv.end() && wwwi != www.end() ; ++vvvi, ++wwwi) { 
    printf("vvv value = %d, index = %ld, www value = %d, index = %ld \n", *vvvi, vvvi-vvv0, *wwwi, wwwi-www0); 
  }
  return 0;
}

#include <cstdio>
#include <vector>
#include <kitsune.h>

int main() {
  
  std::vector<int> vvv{23,24,25,26,27,28,29,30,31,32,33,34,35};

  std::vector<int>::iterator zzz0=vvv.begin();
  forall (std::vector<int>::iterator zzz=zzz0; zzz != vvv.end(); ++zzz) { 
    printf("value = %d, index = %ld\n", *zzz, zzz-zzz0); 
  }
  return 0;
}

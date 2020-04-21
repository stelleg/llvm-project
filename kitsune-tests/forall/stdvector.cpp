#include <cstdio>
#include <vector>
#include <kitsune.h>

int main() {
  std::vector<int> vvv{23,24,25,26,27,28,29,30,31,32,33,34,35};

  forall(int zzz=0; zzz<vvv.size(); ++zzz) { 
    printf("%d\n", vvv[zzz]); 
  }
  return 0;
}

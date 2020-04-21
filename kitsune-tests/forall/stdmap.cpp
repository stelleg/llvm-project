#include <cstdio>
#include <map>
#include <kitsune.h>

int main() {

  std::map<int,int> mmm = {{11,43},{12,44},{13,45},{14,46}};

  forall (std::map<int,int>::iterator mmmi=mmm.begin(); mmmi!=mmm.end(); ++mmmi) { 
    printf("%d:%d\n", mmmi->first, mmmi->second); 
  }

  printf("\n");

  forall (auto&& kv: mmm) { 
    printf("%d:%d\n", kv.first, kv.second); 
  }

  printf("\n");

  forall (auto &[key,value] : mmm){
    printf("%d:%d\n", key, value); 
  }
  
  return 0;
}

#include <cstdio>
#include <vector>
#include <kitsune.h>

int main() {

  std::vector<int> vvv{27,28,29,30,31};

  forall(auto i : vvv) printf("%d\n", i);

  return 0;
}

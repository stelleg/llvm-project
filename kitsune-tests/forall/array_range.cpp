#include <cstdio>
#include <kitsune.h>

int main() {

  int vvv[]={7, 8, 9};

  forall(auto i : vvv) printf("%d\n", i);

  return 0;
}

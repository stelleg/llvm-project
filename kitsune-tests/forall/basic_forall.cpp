#include <cstdio>
#include <kitsune.h>

int main() {
  // 25 is the minimum to avoid inlining
  // int var_before=11;
  forall(int i = 17; i < 25; ++i) {
    // int var_body=12;
    printf("%d\n", i);
  }
  // int var_after=13;
  return 0;
}

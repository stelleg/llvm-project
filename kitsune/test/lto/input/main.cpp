// Multi-file vector addition benchmark. This is purely to test LTO.

#include <iostream>
#include <kitsune.h>

void vecadd(kitsune::mobile_ptr<double> c, const kitsune::mobile_ptr<double> a,
            const kitsune::mobile_ptr<double> b, size_t n);

int main(int argc, char *argv[]) {
  size_t n = atoi(argv[1]);
  unsigned iterations;
  kitsune::mobile_ptr<double> a(n);
  kitsune::mobile_ptr<double> b(n);
  kitsune::mobile_ptr<double> c(n);

  vecadd(c, a, b, n);

  return c[argc];
}

// -----------------------------------------------------------------------------
// Check that the correct warning is emitted when reducing a parallel_reduce
// and that we fall back to the default C++ lowering. Obviously when we support
// parallel_reduce, this test will have to be changed or removed.
//
// RUN: %kitxx -fkokkos -fkokkos-no-init -O1 --tapir=nolo -S -emit-llvm \
// RUN:     -fno-inline -o - %sysroot %s \
// RUN:     | FileCheck %s

#include <Kokkos_Core.hpp>
#include <cstdio>

extern "C" double f(int n) {
  double result;
  // expected-warning-re@+1 {{kokkos - parallel_reduce is not supported{{.*}}}}
  Kokkos::parallel_reduce(
      "sum", n, KOKKOS_LAMBDA(const int &i, double &lsum) { lsum += i; },
      result);

  return result;
}

// CHECK-LABEL: double @f
// CHECK-SAME: %[[N:[^)]+]])
// CHECK: call {{.+}}void @"_ZN6Kokkos15parallel_reduce

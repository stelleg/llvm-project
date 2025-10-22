// -----------------------------------------------------------------------------
// Check that the correct warning is emitted when an unsupported policy is seen
// in a parallel_for and that we fall back to the default C++ lowering.
//
// In this case, we have used a TeamPolicy, but if we ever do support
// TeamPolicy's, this should be replaced with something else that is not
// supported.
//
// RUN: %kitxx -fkokkos -fkokkos-no-init -O1 --tapir=nolo -S -emit-llvm \
// RUN:     -fno-inline -o - %sysroot %s \
// RUN:     | FileCheck %s

#include <Kokkos_Core.hpp>
#include <cstdio>

extern "C" void f(int n) {
  // expected-warning-re@+1 {{kokkos - unsupported expression for range of parallel_for{{.*}}}}
  Kokkos::parallel_for(
      "hello",
      typename Kokkos::TeamPolicy<>(n, Kokkos::AUTO),
      KOKKOS_LAMBDA(
          const typename Kokkos::TeamPolicy<>::member_type &team) {
        printf("Hello world\n");
      });
}

// CHECK-LABEL: void @f
// CHECK-SAME: %[[N:[^)]+]])
// CHECK: call {{.+}}void @"_ZN6Kokkos12parallel_for

// Check that we correctly emit a warning and fall back to standard C++ lowering
// upon encountering a Kokkos::parallel_for with a functor instead of a lambda.
//
// RUN: %kitxx -fkokkos -fkokkos-no-init --tapir=nolo -O1 -S -emit-llvm \
// RUN:     -o - -Xclang -verify %sysroot %s \
// RUN:     | FileCheck -vv %s

#include <Kokkos_Core.hpp>
#include <cstdio>

struct Hello {
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const { printf("Hello %d\n", i); }
};

extern "C" void f(size_t n) {
  // expected-warning-re@+1 {{kokkos - functors not supported in parallel_for{{.*}}}}
  Kokkos::parallel_for(n, Hello());
}

// CHECK-LABEL: void @f
// CHECK-SAME: i64{{.+}} %[[N:[^)]+]]
// CHECK: %[[HELLO:.+]] = alloca %struct.Hello
// CHECK: call void @_ZN6Kokkos12parallel_for{{.+}}(i64{{.+}} %[[N]], ptr{{.+}} %[[HELLO]])

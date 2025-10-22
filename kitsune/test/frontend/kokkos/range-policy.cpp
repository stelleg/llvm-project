// RUN: %kitxx -fkokkos -fkokkos-no-init --tapir=nolo -S -emit-llvm %s \
// RUN:     -O1 -fno-stripmine -fno-vectorize -o - %sysroot \
// RUN:     | FileCheck %s

#include <Kokkos_Core.hpp>

// clang-format off
extern "C" void f(float* a, size_t N) {
  Kokkos::parallel_for("initialize_array", Kokkos::RangePolicy<>(0,N),
                       KOKKOS_LAMBDA(size_t i) {
                         a[i] = 0.0;
                       });
}
// clang-format on

// CHECK-LABEL: void @f
// CHECK-SAME: i64{{.+}} %[[N:.+]])
// CHECK-NEXT: [[ENTRY:.+]]:
// CHECK-NEXT: %[[SYNCREG:.+]] = {{.+}}call token @llvm.syncregion.start
// CHECK: [[DETACH:^.+]]:
// CHECK-NEXT: %[[IV:.+]] = phi i64 [ %[[NEXT:.+]], %[[LATCH:.+]] ], [ 0, %[[ENTRY]] ]
// CHECK-NEXT: detach within %[[SYNCREG]], label %[[BODY:.+]], label %[[LATCH]]
// CHECK: [[BODY]]:
// CHECK: reattach within %[[SYNCREG]], label %[[LATCH]]
// CHECK: [[LATCH]]:
// CHECK-NEXT: %[[NEXT]] = add {{.+}} %[[IV]], 1
// CHECK-NEXT: %[[COND:.+]] = icmp eq i64 %[[NEXT]], %[[N]]
// CHECK-NEXT: br i1 %[[COND]], label %[[SYNC:.+]], label %[[DETACH]]
// CHECK: [[SYNC]]:
// CHECK-NEXT: sync within %[[SYNCREG]]

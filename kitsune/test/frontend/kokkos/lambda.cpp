// -----------------------------------------------------------------------------
// Check that both "anonymous" and "named" parallel_for constructs, given a
// lambda, are lowered correctly
//
// RUN: %kitxx -fkokkos -fkokkos-no-init -O1 --tapir=nolo -S -emit-llvm \
// RUN:     -o - %sysroot %s \
// RUN:     | FileCheck %s

#include <Kokkos_Core.hpp>
#include <cstdio>

extern "C" void anon(int n) {
  // clang-format off
  Kokkos::parallel_for(n, KOKKOS_LAMBDA(int i) {
    printf("hello from %i\n", i);
  });
  // clang-format on
}

// CHECK-LABEL: void @anon
// CHECK-SAME: i32{{.*}} %[[N:[^)]+]]
// CHECK-NEXT: [[ENTRY:.+]]:
// CHECK-NEXT: %[[SYNCREG:.+]] = {{.+}}call token @llvm.syncregion.start
// CHECK: [[DETACH:^.+]]:
// CHECK-NEXT: %[[IV:.+]] = phi i32 [ %[[NEXT:.+]], %[[LATCH:.+]] ], [ 0, %[[ENTRY]] ]
// CHECK-NEXT: detach within %[[SYNCREG]], label %[[BODY:.+]], label %[[LATCH]]
// CHECK: [[BODY]]:
// CHECK-NEXT: call{{.+}} @printf
// CHECK-NEXT: reattach within %[[SYNCREG]], label %[[LATCH]]
// CHECK: [[LATCH]]:
// CHECK-NEXT: %[[NEXT]] = add {{.+}} %[[IV]], 1
// CHECK-NEXT: %[[COND:.+]] = icmp eq i32 %[[NEXT]], %[[N]]
// CHECK-NEXT: br i1 %[[COND]], label %[[SYNC:.+]], label %[[DETACH]]
// CHECK: [[SYNC]]:
// CHECK-NEXT: sync within %[[SYNCREG]]

extern "C" void named(int n) {
  // clang-format off
  Kokkos::parallel_for("name", n, KOKKOS_LAMBDA(int i) {
    printf("hello from %i\n", i);
  });
  // clang-format on
}

// CHECK-LABEL: void @named
// CHECK-SAME: i32{{.*}} %[[N:[^)]+]]
// CHECK-NEXT: [[ENTRY:.+]]:
// CHECK-NEXT: %[[SYNCREG:.+]] = {{.+}}call token @llvm.syncregion.start
// CHECK: [[DETACH:^.+]]:
// CHECK-NEXT: %[[IV:.+]] = phi i32 [ %[[NEXT:.+]], %[[LATCH:.+]] ], [ 0, %[[ENTRY]] ]
// CHECK-NEXT: detach within %[[SYNCREG]], label %[[BODY:.+]], label %[[LATCH]]
// CHECK: [[BODY]]:
// CHECK-NEXT: call{{.+}} @printf
// CHECK-NEXT: reattach within %[[SYNCREG]], label %[[LATCH]]
// CHECK: [[LATCH]]:
// CHECK-NEXT: %[[NEXT]] = add {{.+}} %[[IV]], 1
// CHECK-NEXT: %[[COND:.+]] = icmp eq i32 %[[NEXT]], %[[N]]
// CHECK-NEXT: br i1 %[[COND]], label %[[SYNC:.+]], label %[[DETACH]]
// CHECK: [[SYNC]]:
// CHECK-NEXT: sync within %[[SYNCREG]]

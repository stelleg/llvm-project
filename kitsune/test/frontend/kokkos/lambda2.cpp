// Check that multiple Kokkos parallel_for loops in the same function are
// handled correctly. The fact that one is anonymous and one is named is not
// relevant
//
// RUN: %kitxx -fkokkos -fkokkos-no-init -O1 --tapir=nolo -S -emit-llvm \
// RUN:     -o - %sysroot %s \
// RUN:     | FileCheck %s

#include <Kokkos_Core.hpp>
#include <cstdio>

extern "C" void f(int n) {
  // clang-format off
  Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
    printf("hello from %i\n", i);
  });

  Kokkos::parallel_for("hello", n, KOKKOS_LAMBDA(const int i) {
    printf("hello from %i\n", i);
  });
  // clang-format on
}

// CHECK-LABEL: void @f
// CHECK-SAME: i32{{.*}} %[[N:[^)]+]]
// CHECK-NEXT: [[ENTRY:.+]]:
// CHECK-NEXT: %[[SYNCREG:.+]] = {{.+}}call token @llvm.syncregion.start
// CHECK: [[LOOP1:^.+]]:
// CHECK-NEXT: %[[IV1:.+]] = phi i32 [ %[[NEXT1:.+]], %[[LATCH1:.+]] ], [ 0, %[[ENTRY]] ]
// CHECK-NEXT: detach within %[[SYNCREG]], label %[[BODY1:.+]], label %[[LATCH1]]
// CHECK: [[BODY1]]:
// CHECK-NEXT: call{{.+}} @printf
// CHECK-NEXT: reattach within %[[SYNCREG]], label %[[LATCH1]]
// CHECK: [[LATCH1]]:
// CHECK-NEXT: %[[NEXT1]] = add {{.+}} %[[IV1]], 1
// CHECK-NEXT: %[[COND1:.+]] = icmp eq i32 %[[NEXT1]], %[[N]]
// CHECK-NEXT: br i1 %[[COND1]], label %[[SYNC1:.+]], label %[[LOOP1]]
// CHECK: [[SYNC1]]:
// CHECK-NEXT: sync within %[[SYNCREG]]
// CHECK: [[LOOPPH2:^.+]]:
// CHECK: [[LOOP2:^.+]]:
// CHECK-NEXT: %[[IV2:.+]] = phi i32 [ %[[NEXT2:.+]], %[[LATCH2:.+]] ], [ 0, %[[LOOPPH2]] ]
// CHECK-NEXT: detach within %[[SYNCREG]], label %[[BODY2:.+]], label %[[LATCH2]]
// CHECK: [[BODY2]]:
// CHECK-NEXT: call{{.+}} @printf
// CHECK-NEXT: reattach within %[[SYNCREG]], label %[[LATCH2]]
// CHECK: [[LATCH2]]:
// CHECK-NEXT: %[[NEXT2]] = add {{.+}} %[[IV2]], 1
// CHECK-NEXT: %[[COND2:.+]] = icmp eq i32 %[[NEXT2]], %[[N]]
// CHECK-NEXT: br i1 %[[COND2]], label %[[SYNC2:.+]], label %[[LOOP2]]
// CHECK: [[SYNC2]]:
// CHECK-NEXT: sync within %[[SYNCREG]]

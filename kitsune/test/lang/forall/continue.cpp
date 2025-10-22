// RUN: %kitxx --tapir=nolo -S -emit-llvm -o - %s | FileCheck %s

#include <kitsune.h>

void loop(int *a) {
  // continue statements are allowed in a forall.
  forall(int i = 0; i < 10; i++) {
    if (i == 4) {
      continue;
    }
    a[i] = i;
  }
}

// CHECK: detach within %[[SYNCREG:.+]], label {{.+}}, label {{.+}}
// CHECK: %[[CMP:.+]] = icmp eq {{.+}}, 4
// CHECK: br i1 %[[CMP]], label %[[THEN:.+]], label %[[ELSE:.+]]
// CHECK: [[THEN]]:
// CHECK-NEXT: br label %[[REATTACH:.+]]
// CHECK: [[ELSE]]:
// CHECK: store
// CHECK-NEXT: br label %[[REATTACH]]
// CHECK: [[REATTACH]]:
// CHECK-NEXT: reattach within %[[SYNCREG]]

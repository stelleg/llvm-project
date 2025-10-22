; Check that arguments to kernels that are pointers are handled correctly.
; Simply setting these arguments to be in the correct address space is not
; sufficient since we need to ensure that the change in address space does not
; affect the instructions in the body. For instance, if the pointer argument is
; passed to a device function, there may be a type mismatch if the device
; function does not expect pointers in a specific address space. In order to
; deal with this, we cast away the address space early in the kernel function.
; This checks that this casting is introduced in the correct spot - the entry
; block of the function, after the allocas, and is propagated to the uses of the
; arguments in the body of the function.
;
; RUN: opt --tapir=hip %s --tapir-hip-features="+16-bit-insts" \
; RUN:     -passes='loop-spawning,emb-prepare' \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: define {{.+}} @__kithip_loop_f{{[^(]*}}(
; CHECK-SAME: ptr addrspace(1) align 1 %[[A:[^,]+]],
; CHECK-SAME: ptr addrspace(1) align 1 %[[B:[^,]+]],
; CHECK-SAME: ptr addrspace(1) align 1 %[[C:[^)]+]])
; CHECK: %[[CSTA:.+]] = addrspacecast ptr addrspace(1) %[[A]] to ptr
; CHECK: %[[CSTB:.+]] = addrspacecast ptr addrspace(1) %[[B]] to ptr
; CHECK: %[[CSTC:.+]] = addrspacecast ptr addrspace(1) %[[C]] to ptr
; CHECK: %[[IV:.+]] = phi i64
; CHECK: %[[V0:.+]] = tail call fastcc ptr @id(ptr %[[CSTA]])
; CHECK: %[[V1:.+]] = ptrtoint ptr %[[CSTB]] to i64
; CHECK: %[[V2:.+]] = getelementptr inbounds ptr, ptr %[[CSTC]], i64 %[[IV]]
; CHECK: %[[V3:.+]] = load i64, ptr %[[CSTA]]
; CHECK: store ptr %[[V2]], ptr %[[CSTB]]
; CHECK: %[[V4:.+]] = add i64 %[[V1]], %[[V3]]
; CHECK: store i64 %[[V4]], ptr %[[V2]]

target triple = "x86_64-pc-linux-gnu"

define ptr @id(ptr %p) {
  ret ptr %p
}

define dso_local void @f(ptr %a, ptr %b, ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp5 = icmp sgt i64 %n, 0
  br i1 %cmp5, label %preheader, label %forall.sync

preheader:
  br label %forall.detach

forall.detach:
  %indvars.iv = phi i64 [ 0, %preheader ], [ %indvars.iv.next, %forall.inc ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:
  %0 = tail call ptr @id(ptr %a)
  %1 = ptrtoint ptr %b to i64
  %2 = getelementptr inbounds ptr, ptr %c, i64 %indvars.iv
  %3 = load i64, ptr %a
  store ptr %2, ptr %b
  %4 = add i64 %1, %3
  store i64 %4, ptr %2
  reattach within %syncreg, label %forall.inc

forall.inc:
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !0

forall.sync:
  sync within %syncreg, label %forall.end

forall.end:
  ret void
}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"llvm.loop.unroll.disable"}

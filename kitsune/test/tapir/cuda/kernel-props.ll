; Check that the kernel properties pass updates the initializer of the kernel
; properties global variable.
;
; At the time of writing this test, the only properties that are computed are
; the instruction mix i.e. the counts of various instruction kinds used in the
; kernel. The first two elements of the properties struct are the number of
; memory operations and the number of floating point operations. We only check
; for these since this particular kernel has been crafted such that those values
; can be computed easily. If we change the kernel properties that are computed,
; this test, and the type of the global variable will have to be udpated.
;
; RUN: opt --tapir=cuda --tapir-cuda-arch=sm_80 -S %s \
; RUN:     -passes='tapir-lowering<O2>,kit-kernel-properties' \
; RUN:     | FileCheck %s
;
; CHECK: @{{.+}} = private unnamed_addr constant
; CHECK-SAME: { i64, i64, i64, i64 }
; CHECK-SAME: { i64 2, i64 1, {{.+}} }
; CHECK-SAME: #[[KERNEL_PROPS:[0-9]+]]
;
; CHECK: attributes #[[KERNEL_PROPS]] = {
; CHECK-SAME: "kit_kernel_props"="__kitcu_loop_{{.+}}"

target triple = "x86_64-pc-linux-gnu"

define void @f(ptr %c, i64 %n) {
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
  %arrayidx = getelementptr inbounds float, ptr %c, i64 %indvars.iv
  %v = load float, ptr %arrayidx, align 4
  %v2 = fmul float %v, %v
  store float %v2, ptr %arrayidx, align 4
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
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"llvm.loop.unroll.disable"}

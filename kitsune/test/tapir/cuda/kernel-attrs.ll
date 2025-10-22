; Check that the correct attributes have been added to and removed from the
; kernel function.
;
; NOTE: We don't yet fully understand which attributes are actually needed for
; correctness or beneficial for performance. For the immediate future, the
; checks in this test will have to be updated to correctly reflect what CudaABI
; does.
;
; RUN: opt --tapir=cuda --tapir-cuda-arch=sm_72 --tapir-cuda-features="+ptx87" \
; RUN:     %s -passes='tapir-lowering<O2>' \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: define {{.*}}ptx_kernel void {{.+}} #[[ATTRS:[0-9]+]]
; CHECK: attributes #[[ATTRS]] = {
; CHECK-NOT: "personality"
; CHECK-NOT: "tune-cpu"
; CHECK-SAME: kit_kernel
; CHECK-SAME: "target-cpu"="sm_72"
; CHECK-SAME: "target-features"="+ptx87,sm_72"
; CHECK-SAME: "uniform-work-group-size"="true"

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
  %arrayidx = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  store i64 %n, ptr %arrayidx, align 4
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

; Check that the correct attributes have been added to and removed from the
; kernel function
;
; NOTE: We don't yet fully understand which attributes are actually needed for
; correctness or beneficial for performance. For the immediate future, the
; checks in this test will have to be updated to correctly reflect what HipABI
; does.
;
; RUN: opt --tapir=hip \
; RUN:     --tapir-hip-arch=gfx906 \
; RUN:     --tapir-hip-features="+wavefrontsize32,+atomic-fadd-rtn-insts" \
; RUN:     -passes='tapir-lowering<O2>' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; The visibility and calling convention must be set.
;
; CHECK: define protected amdgpu_kernel {{.+}} #[[ATTRS:[0-9]+]]
; CHECK: attributes #[[ATTRS]] = {
; CHECK-NOT: "personality"
; CHECK-NOT: "tune-cpu"
; CHECK-NOT: "uwtable"
; CHECK-SAME: kit_kernel
; CHECK-SAME: mustprogress
; CHECK-SAME: nounwind
; CHECK-SAME: "amdgpu-flat-work-group-size"="128,1024"
; CHECK-SAME: "no-trapping-math"="true"
; CHECK-SAME: "target-cpu"="gfx906"
; CHECK-SAME: "target-features"="+wavefrontsize32,+atomic-fadd-rtn-insts"
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
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"llvm.loop.unroll.disable"}

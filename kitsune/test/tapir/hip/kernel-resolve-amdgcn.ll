; Check that amdgcn intrinsics are resolved correctly. This only checks for a
; few intrinsics which is not terribly useful in the grand scheme of things.
;
; RUN: opt --tapir=hip --tapir-hip-runtime-bcs=%S/input/libdevice.ll %s \
; RUN:     -passes='tapir-lowering<O2>,emb-resolve-libdevice-calls' \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: %[[ASF:.+]] = uitofp nneg i64
; CHECK-NEXT: %[[SQRT:.+]] = tail call double @llvm.amdgcn.rsq.f64(double %[[ASF]])
; CHECK-NEXT: store double %[[SQRT]],

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
  %.cst = sitofp i64 %indvars.iv to double
  %.rsq = tail call double @llvm.amdgcn.rsq.f64(double %.cst)
  store double %.rsq, ptr %arrayidx, align 4
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

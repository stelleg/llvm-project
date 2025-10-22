; Check that the command line option to force inline all device functions
; (including those that have the noinline attribute) is handled correctly.
;
; RUN: opt --tapir=cuda --tapir-cuda-arch=sm_72 %s \
; RUN:     -passes='tapir-lowering<O2>,emb-prepare' -emb-inline-all-force \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: define {{.+}} @device_func{{.+}} #[[ATTRS:[0-9]+]]
; CHECK: attributes #[[ATTRS]] = {
; CHECK-SAME: alwaysinline
; CHECK-NOT: noinline

target triple = "x86_64-pc-linux-gnu"

define i64 @device_func(i64 %n) {
  ret i64 %n
}

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
  %.call = call i64 @device_func(i64 %n)
  store i64 %.call, ptr %arrayidx, align 4
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

; Check that the prefetch pass inserts host-to-device prefetch calls correctly.
; Currently, we do not compute the number of bytes to be prefetched and always
; pass -1 indicating that the runtime should compute the number of bytes to be
; prefetched. If this changes, this test must be updated.
;
; RUN: opt --tapir=hip -passes='tapir-lowering<O2>,kit-prefetch' -S %s \
; RUN:     --tapir-gpu-prefetch=true \
; RUN:     | FileCheck %s
;
; CHECK: define {{.+}} @f
; CHECK: %[[STREAM:[0-9]+]] = {{.*}}call ptr @llvm.kit.thread.stream(i32 4)
; CHECK: call {{.+}} @llvm.kit.async.prefetch.htod(i32 4, ptr %c, i64 -1, ptr %[[STREAM]])
; CHECK: call {{.+}} @llvm.kit.async.launch.kernel(i32 4,
;
; -----------------------------------------------------------------------------

target triple = "x86_64-unknown-linux-gnu"

define void @f(ptr %c, float %scale, i64 %n) {
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
  %scaled = fmul float %v, %scale
  store float %scaled, ptr %arrayidx, align 4
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

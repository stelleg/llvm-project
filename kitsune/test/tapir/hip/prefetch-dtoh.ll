; Check that the prefetch pass inserts device-to-host prefetch calls correctly.
;
; Currently, we do not insert such prefetch calls, so the checks here ensure
; that this call is not inserted. The test code itself is crafted to ensure that
; the array is accessed on the host after the forall loop, so when we do
; implement device-to-host prefetches, one is likely to be inserted. When we do
; implement this, this comment should be updated/removed.
;
; RUN: opt --tapir=hip -passes='tapir-lowering<O2>,kit-prefetch' -S %s \
; RUN:     --tapir-gpu-prefetch=true \
; RUN:     | FileCheck %s
;
; CHECK: define {{.+}} @f
; CHECK: %[[STREAM:[0-9]+]] = {{.*}}call ptr @llvm.kit.thread.stream(i32 4)
; CHECK: call {{.+}} @llvm.kit.async.launch.kernel(i32 4
; CHECK-NOT: call {{.+}} @llvm.kit.async.prefetch.dtoh(i32 4
;
; -----------------------------------------------------------------------------

target triple = "x86_64-unknown-linux-gnu"

declare void @printf32(float)

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
  %postidx = getelementptr inbounds float, ptr %c, i64 %n
  %w = load float, ptr %postidx, align 4
  call void @printf32(float %w)
  br label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"llvm.loop.unroll.disable"}

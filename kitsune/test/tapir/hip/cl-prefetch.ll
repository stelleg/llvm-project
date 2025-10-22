; Check that the --tapir-gpu-prefetch option is handled correctly
;
; -----------------------------------------------------------------------------
;
; RUN: opt --tapir=hip -passes='tapir-lowering<O2>,kit-prefetch' -S %s \
; RUN:     --tapir-gpu-prefetch=true \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     | FileCheck %s -check-prefix PREFETCH
;
; PREFETCH: define {{.+}} @f
; PREFETCH: call {{.+}} @llvm.kit.async.prefetch.htod(i32 4,
; PREFETCH: call {{.+}} @llvm.kit.async.launch.kernel(i32 4,
; PREFETCH: ret void
; PREFETCH-NEXT: }
;
; -----------------------------------------------------------------------------
;
; RUN: opt --tapir=hip -passes='tapir-lowering<O2>,kit-prefetch' -S %s \
; RUN:     --tapir-gpu-prefetch=false \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     | FileCheck %s -check-prefix NO-PREFETCH
;
; NO-PREFETCH: define {{.+}} @f
; NO-PREFETCH-NOT: call {{.+}} @llvm.kit.async.prefetch.htod(i32 4,
; NO-PREFETCH: call {{.+}} @llvm.kit.async.launch.kernel(i32 4,
; NO-PREFETCH: ret void
; NO-PREFETCH-NEXT: }
;
; -----------------------------------------------------------------------------

target triple = "x86_64-unknown-linux-gnu"

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
  %arrayidx = getelementptr inbounds i64, ptr %c, i64 %indvars.iv
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

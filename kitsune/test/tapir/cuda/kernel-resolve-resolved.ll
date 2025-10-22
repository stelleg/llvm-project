; Check that functions with the __nv and __nv_fast prefixes are left as is.
;
; RUN: opt --tapir=cuda --tapir-cuda-runtime-bc=%S/input/libdevice.ll %s \
; RUN:     -passes='tapir-lowering<O2>,emb-resolve-libdevice-calls' \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: %[[ASF:.+]] = uitofp nneg i64
; CHECK-NEXT: %[[SIN:.+]] = tail call float @__nv_sinf(float %[[ASF]])
; CHECK-NEXT: %[[SINF:.+]] = tail call float @__nv_fast_sinf(float %[[SIN]])
; CHECK-NEXT: store float %[[SINF]],

target triple = "x86_64-pc-linux-gnu"

declare float @__nv_sinf(float)
declare float @__nv_fast_sinf(float)

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
  %asf = sitofp i64 %indvars.iv to float
  %sin = tail call float @__nv_sinf(float %asf)
  %sinf = tail call float @__nv_fast_sinf(float %sin)
  store float %sinf, ptr %arrayidx, align 4
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

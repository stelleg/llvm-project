; Check that linking libdevice bitcode works as expected.
;
; RUN: opt --tapir=cuda --tapir-cuda-runtime-bc=%S/input/libdevice.ll %s \
; RUN:     -passes='tapir-lowering<O2>,emb-resolve-libdevice-calls,emb-link-libdevice-bitcode' \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; We should only link in what is actually needed
; CHECK-NOT: define {{.+}} @__nv_fast_sinf
; CHECK-NOT: define {{.+}} @__nv_cosf
;
; CHECK-LABEL: define {{.+}} @__kitcu_
; CHECK: tail call float @__nv_sinf
; CHECK: tail call fast float @__nv_fast_cosf
; CHECK-DAG: define {{.+}} @__nv_sinf
; CHECK-DAG: define {{.+}} @__nv_fast_cosf

target triple = "x86_64-pc-linux-gnu"

declare float @sinf(float)
declare float @cosf(float)

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
  %asf = sitofp i64 %n to float
  %sin = tail call float @sinf(float %asf)
  %cos = tail call fast float @cosf(float %sin)
  %asi = fptosi float %cos to i64
  store i64 %asi, ptr %arrayidx, align 4
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

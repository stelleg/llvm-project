; When device functions are resolved, a declaration for the libdevice function
; is added to the device module, even if a definition is provided by the
; libdevice module. The emb-link-libdevice-bitcode pass will provide the definition.
; When doing this, we need to change the linkage types of the libdevice
; functions because only external and external_weak linkage is allowed on
; declarations.
;
; RUN: opt --tapir=hip --tapir-hip-runtime-bcs=%S/input/libdevice.ll %s \
; RUN:     -passes='tapir-lowering<O2>,emb-resolve-libdevice-calls' \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: declare {{.*}}float @__ocml_acos_f32
; CHECK: declare {{.*}}double @__ocml_sqrt_f64

target triple = "x86_64-pc-linux-gnu"

declare float @acosf(float)
declare double @sqrt(double)

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
  %asf = sitofp i64 %n to float
  %.acos = tail call float @acosf(float %asf)
  %.cst = fpext float %.acos to double
  %.sqrt = tail call double @sqrt(double %.cst)
  store double %.sqrt, ptr %arrayidx, align 4
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

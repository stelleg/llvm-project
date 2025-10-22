; Check that the loop bounds of the tapir loop are correctly replaced in the
; outlined kernel function when a 1D kernel is launched.
;
; NOTE: Currently, we use a grainsize of 1 i.e. every thread computes a single
; iteration of the tapir loop, but if that changes, this test must be updated.
;
; RUN: opt --tapir=cuda --tapir-cuda-arch="sm_72" %s \
; RUN:     -passes='tapir-lowering<O2>' \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: define {{.+}}(i64 {{.*}}%[[UB:[^,]+]], i64 {{[^,]+}}, i64 {{[^,]+}}, ptr {{.*}}%[[BUF:[^,]+]], i64 {{.*}}%[[N:[^)]+]]) #[[ATTRS:[0-9]+]]
; CHECK-NEXT: [[PREHEADER:.+]]:
; CHECK: %[[TID:.+]] = {{.*}}call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK: %[[BIDX:.+]] = {{.*}}call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
; CHECK: %[[BDIM:.+]] = {{.*}}call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
; CHECK: %[[BOFF:.+]] = mul i32 %[[BIDX]], %[[BDIM]]
; CHECK: %[[IV32:.+]] = add i32 %[[TID]], %[[BOFF]]
; CHECK: %[[TIV:.+]] = zext i32 %[[IV32]] to i64
; CHECK: %[[TEND:.+]] = add i64 %[[TIV]], 1
;
; CHECK: [[HEADER:.+]]:
; CHECK: %[[IV:.+]] = phi i64 [ %[[IV_NEXT:.+]], %[[LATCH:.+]] ], [ %[[TIV]], %[[PREHEADER]] ]
; CHECK: %[[IV_NEXT:.+]] = add {{.*}}i64 %[[IV]], 1
; CHECK: %[[COND:.+]] = icmp eq i64 %[[IV_NEXT]], %[[TEND]]
; CHECK: br i1 %[[COND]], label %[[BBEXIT:.+]], label %[[HEADER]]
;
; CHECK: [[BBEXIT]]:
; CHECK-NEXT: ret void
;
; CHECK: attributes #[[ATTRS]] = {
; CHECK-SAME: kit_kernel

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

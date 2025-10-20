; Check that the induction variable for a 1D launch is computed correctly and
; bypasses the body of the loop if out of bounds
;
; RUN: opt --tapir=cuda --tapir-cuda-arch="sm_72" %s \
; RUN:     -passes='tapir-lowering<O2>' \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: define {{[^(]+}}(
; CHECK-SAME: i64 {{[^%]*}}%[[LB:[^,]+]],
; CHECK-SAME: i64 {{[^%]*}}%[[UB:[^,]+]],
; CHECK-SAME: i64 {{[^)]+}})
; CHECK-SAME: #[[ATTRS:[0-9]+]]
; CHECK: %[[TID:.+]] = {{.*}}call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK: %[[BIDX:.+]] = {{.*}}call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
; CHECK: %[[BDIM:.+]] = {{.*}}call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
; CHECK: %[[BOFF:.+]] = mul i32 %[[BIDX]], %[[BDIM]]
; CHECK: %[[IV32:.+]] = add i32 %[[TID]], %[[BOFF]]
; CHECK: %[[TIV:.+]] = zext i32 %[[IV32]] to i64
; CHECK: %[[COND:.+]] = icmp uge i64 %[[TIV]], %[[UB]]
; CHECK-NEXT: br i1 %[[COND]], label %[[BBEXIT:[^,]+]],
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

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}

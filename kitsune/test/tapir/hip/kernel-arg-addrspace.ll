; Check that the parameters of the outlined kernel function are in the correct
; address space. The arguments to the device functions do not need to be in any
; particular address space.
;
; RUN: opt --tapir=hip %s --tapir-hip-features="+16-bit-insts" \
; RUN:     -passes='tapir-lowering<O2>,emb-prepare' \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: define {{.+}} @id(
; CHECK-SAME: ptr %{{.+}})
;
; CHECK: define {{.+}} @__kithip_loop_f{{[^(]*}}(
; CHECK-SAME: ptr addrspace(1) align 1 %{{[^,]+}},
; CHECK-SAME: ptr addrspace(1) align 1 %{{[^,]+}},
; CHECK-SAME: ptr addrspace(1) align 1 %{{[^)]+}})

target triple = "x86_64-pc-linux-gnu"

define ptr @id(ptr %p) {
  ret ptr %p
}

define void @f(ptr %c, ptr %a, ptr %b, i64 %n) {
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
  %ptra = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %0 = load i32, ptr %ptra
  %ptrb = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %1 = load i32, ptr %ptrb
  %2 = add i32 %0, %1
  %3 = inttoptr i32 %2 to ptr
  %4 = tail call ptr @id(ptr %3)
  %ptrc = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  store ptr %4, ptr %ptrc, align 4
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

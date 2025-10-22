; Check that calls to printf are lowered correctly.
;
; RUN: opt --tapir=hip --tapir-hip-runtime-bcs=%S/input/libdevice.ll %s \
; RUN:     -passes='tapir-lowering<O2>,emb-resolve-libdevice-calls' \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: %[[IV:.+]] = phi i64
; CHECK: %[[PRINTF_ALLOC_FN:.+]] = call ptr addrspace(1) @__printf_alloc
; CHECK: %[[COND:.+]] = icmp ne ptr addrspace(1) %[[PRINTF_ALLOC_FN]], null
; CHECK: br i1 %[[COND]], label %[[BBARGS:.+]], label %[[BBEND:.+]]
; CHECK: [[BBBODY:.+]]:
; CHECK: br label %[[BBINC:.+]]
; CHECK: [[BBINC]]:
; CHECK: [[BBEND]]:
; CHECK: br label %[[BBBODY]]
; CHECK: [[BBARGS]]:
; CHECK: store i32 {{[0-9]+}}, ptr addrspace(1) %[[PRINTF_ALLOC_FN]]
; CHECK: br label %[[BBEND]]

target triple = "x86_64-pc-linux-gnu"

@.str = private unnamed_addr constant [9 x i8] c"%ld: %s\0A\00", align 1
@.str.1 = private unnamed_addr constant [12 x i8] c"Hello world\00", align 1

declare i32 @printf(ptr, ...)

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
  %call7 = call i32 (ptr, ...) @printf(ptr @.str, i64 %indvars.iv, ptr @.str.1)
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

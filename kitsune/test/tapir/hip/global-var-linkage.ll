; Check that any non-constant global variables have external linkage in the
; kernel module, regardless of their linkage in the host.
;
; RUN: opt --tapir=hip -passes='loop-spawning' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; If the variables have external linkage, an explicit linkage will not appear
; here. But the visibility and address space will. But we don't check for those
; explicitly to avoid test failures if either of those properties are changed.
; A separate test will explicitly look for the visibility
;
; CHECK-DAG: @v137 = {{[a-z_]+}} {{[A-Za-z0-9()]+}} global i32
; CHECK-DAG: @v138 = {{[a-z_]+}} {{[A-Za-z0-9()]+}} global i32
; CHECK-DAG: @v139 = {{[a-z_]+}} {{[A-Za-z0-9()]+}} global i32

target triple = "x86_64-unknown-linux-gnu"

@v137 = global i32 13, align 4
@v138 = external global i32, align 4
@v139 = internal global i32 291, align 4

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp4.not = icmp eq i64 %n, 0
  br i1 %cmp4.not, label %forall.sync, label %forall.ph

forall.ph:
  br label %forall.detach

forall.detach:
  %i.05 = phi i64 [ %inc, %forall.inc ], [ 0, %forall.ph ]
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:
  %0 = load i32, ptr @v137, align 4
  %1 = load i32, ptr @v138, align 4
  %2 = add nsw i32 %0, %1
  %3 = load i32, ptr @v139, align 4
  %4 = add nsw i32 %2, %3
  %arrayidx = getelementptr inbounds i32, ptr %c, i64 %i.05
  store i32 %4, ptr %arrayidx, align 4
  reattach within %syncreg, label %forall.inc

forall.inc:
  %inc = add nuw i64 %i.05, 1
  %exitcond.not = icmp eq i64 %inc, %n
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

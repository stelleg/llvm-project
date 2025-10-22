; Check that any constant global variables are copied into the kernel module
; but with the linkage set to internal, regardless of what the linkage is in
; the host module
;
; RUN: opt --tapir=cuda -passes='loop-spawning' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-DAG: @v137 = internal constant i32 921
; CHECK-DAG: @v138 = internal constant i32 11
; CHECK-DAG: @v139 = internal constant i32 46

target triple = "x86_64-unknown-linux-gnu"

@v137 = constant i32 921, align 4
@v138 = private constant i32 11, align 4
@v139 = internal constant i32 46, align 4

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
  %2 = add nuw i32 %0, %1
  %3 = load i32, ptr @v139, align 4
  %4 = add nuw i32 %2, %3
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
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"llvm.loop.unroll.disable"}

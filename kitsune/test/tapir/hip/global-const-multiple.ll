; Check that if the same constant global is used in two separate tapir loops,
; only a single instance of the global is created in the kernel module.
;
; RUN: opt --tapir=hip -passes='tapir-lowering<O2>' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-DAG: @v137{{[^ ]*}} = internal addrspace(4) constant [4 x i32]
; CHECK-DAG: @v138{{[^ ]*}} = internal addrspace(4) constant [4 x i32]
; CHECK-NOT: @v137{{.*}} =
; CHECK-NOT: @v138{{.*}} =

target triple = "x86_64-unknown-linux-gnu"

@v137 = internal constant [4 x i32] [i32 991, i32 0, i32 11, i32 97], align 4
@v138 = private constant [4 x i32] [i32 13, i32 17, i32 91, i32 23], align 4

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp4.not = icmp eq i64 %n, 0
  br i1 %cmp4.not, label %forall.sync, label %forall.detach

forall.detach:
  %i.05 = phi i64 [ %inc, %forall.inc ], [ 0, %entry ]
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:
  %0 = getelementptr inbounds i32, ptr @v137, i64 %i.05
  %1 = load i32, ptr %0, align 4
  %2 = getelementptr inbounds i32, ptr @v138, i64 %i.05
  %3 = load i32, ptr %2, align 4
  %4 = add i32 %1, %3
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

define void @g(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp4.not = icmp eq i64 %n, 0
  br i1 %cmp4.not, label %forall.sync, label %forall.detach

forall.detach:
  %i.05 = phi i64 [ %inc, %forall.inc ], [ 0, %entry ]
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:
  %0 = getelementptr inbounds i32, ptr @v137, i64 %i.05
  %1 = load i32, ptr %0, align 4
  %2 = getelementptr inbounds i32, ptr @v138, i64 %i.05
  %3 = load i32, ptr %2, align 4
  %4 = sub i32 %1, %3
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

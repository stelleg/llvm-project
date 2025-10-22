; Check that the names of global variables are fixed before generating the
; fat binary. ptxas does not allow names containing "."'s which can be present
; in code that is outlined into the kernel module, especially in templated C++
; code.
;
; RUN: opt --tapir=cuda -passes='tapir-lowering<O2>' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-DAG: @__kitcu__nwnm__v137_suffix = {{.*}}global i32
; CHECK-DAG: @__kitcu__nwnm__v138_const = internal constant [4 x i32]

target triple = "x86_64-unknown-linux-gnu"

@v137.suffix = external global i32, align 4
@v138.const = constant [4 x i32] [i32 10, i32 21, i32 42, i32 93]

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp4.not = icmp eq i64 %n, 0
  br i1 %cmp4.not, label %forall.sync, label %forall.detach

forall.detach:
  %i.05 = phi i64 [ %inc, %forall.inc ], [ 0, %entry ]
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:
  %0 = load i32, ptr @v137.suffix, align 4
  %1 = getelementptr inbounds i32, ptr @v138.const, i64 %i.05
  %2 = load i32, ptr %1, align 4
  %3 = add nuw i32 %0, %2
  %arrayidx = getelementptr inbounds i32, ptr %c, i64 %i.05
  store i32 %3, ptr %arrayidx, align 4
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

; Check that the names of the outlined kernel functions are as expected. This
; contains both mangled and demangled function names. This checks that the names
; are demangled when generating the outlined kernel name.
;
; RUN: opt --tapir=hip %s \
; RUN:     -passes='tapir-lowering<O2>' \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-DAG: define {{.+}} @__kithip_loop_scale_0(
; CHECK-DAG: define {{.+}} @__kithip_loop_xlate_1(
; CHECK-DAG: define {{.+}} @__kithip_loop_xlate_2(

target triple = "x86_64-unknown-linux-gnu"

define void @_Z5scalePffm(ptr %buf, float %factor, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp4.not = icmp eq i64 %n, 0
  br i1 %cmp4.not, label %forall.sync, label %forall.detach

forall.detach:
  %i.05 = phi i64 [ %inc, %forall.inc ], [ 0, %entry ]
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:
  %arrayidx = getelementptr inbounds float, ptr %buf, i64 %i.05
  %0 = load float, ptr %arrayidx, align 4
  %mul = fmul float %factor, %0
  store float %mul, ptr %arrayidx, align 4
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

define void @xlate(ptr %buf, float %dist, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp4.not = icmp eq i64 %n, 0
  br i1 %cmp4.not, label %forall.sync, label %forall.detach

forall.detach:
  %i.05 = phi i64 [ %inc, %forall.inc ], [ 0, %entry ]
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:
  %arrayidx = getelementptr inbounds float, ptr %buf, i64 %i.05
  %0 = load float, ptr %arrayidx, align 4
  %add = fadd float %dist, %0
  store float %add, ptr %arrayidx, align 4
  reattach within %syncreg, label %forall.inc

forall.inc:
  %inc = add nuw i64 %i.05, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !4

forall.sync:
  sync within %syncreg, label %forall.ph.2

forall.ph.2:
  %syncreg.2 = tail call token @llvm.syncregion.start()
  %cmp4.not.2 = icmp eq i64 %n, 0
  br i1 %cmp4.not.2, label %forall.sync.2, label %forall.detach.2

forall.detach.2:
  %i.06 = phi i64 [ %inc.2, %forall.inc.2 ], [ 0, %forall.ph.2 ]
  detach within %syncreg.2, label %forall.body.2, label %forall.inc.2

forall.body.2:
  %arrayidx.2 = getelementptr inbounds float, ptr %buf, i64 %i.06
  %1 = load float, ptr %arrayidx.2, align 4
  %add.2 = fadd float %dist, %1
  store float %add.2, ptr %arrayidx.2, align 4
  reattach within %syncreg.2, label %forall.inc.2

forall.inc.2:
  %inc.2 = add nuw i64 %i.06, 1
  %exitcond.not.2 = icmp eq i64 %inc.2, %n
  br i1 %exitcond.not.2, label %forall.sync.2, label %forall.detach.2, !llvm.loop !5

forall.sync.2:
  sync within %syncreg.2, label %forall.end

forall.end:
  ret void
}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"llvm.loop.unroll.disable"}
!4 = distinct !{!4, !1, !2, !3}
!5 = distinct !{!5, !1, !2, !3}

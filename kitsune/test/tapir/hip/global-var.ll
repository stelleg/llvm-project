; Check that any non-const global variables are handled correctly when
; generating kernel launch calls. They should be memcpy'ed to and from the
; before and after launch calls and must be registered with the runtime in
; the ctor for kitsune's runtime.
;
; RUN: opt --tapir=hip -passes='tapir-lowering<O2>,kit-ctors' \
; RUN:     -S %s \
; RUN:     | FileCheck %s
;
; CHECK-DAG: @[[FB:.+]] = constant {{.+}} #[[ATTR:[0-9]+]]
; CHECK-DAG: @[[HOSTVAR:.+]] = external {{.+}} i32
; CHECK-DAG: @[[VARNAME:.+]] = private unnamed_addr constant [5 x i8] c"v137\00"
;
; CHECK: define {{.+}} @f
; CHECK: %[[PTR1:.+]] = {{.*}}call {{.+}} @llvm.kit.symbol.device.ptr(i32 4, ptr nonnull @[[FB]], ptr nonnull @[[VARNAME]])
; CHECK: call {{.+}} @llvm.kit.symbol.memcpy.htod(i32 4, ptr %[[PTR1]], ptr nonnull @[[HOSTVAR]], i64 4)
; CHECK: %[[TS:.+]] = {{.*}}call {{.+}} @llvm.kit.async.launch.kernel(i32 4, ptr nonnull @[[FB]],
; CHECK: %[[PTR2:.+]] = {{.*}}call {{.+}} @llvm.kit.symbol.device.ptr(i32 4, ptr nonnull @[[FB]], ptr nonnull @[[VARNAME]])
; CHECK: call {{.+}} @llvm.kit.symbol.memcpy.dtoh(i32 4, ptr nonnull @[[HOSTVAR]], ptr %[[PTR2]], i64 4)
; CHECK: ret void
; CHECK-NEXT: }
;
; CHECK: define {{.+}} @.kithip.ctor{{[^(]*}}
; CHECK: %[[HANDLE:.+]] = call ptr @__hipRegisterFatBinary
; CHECK: call {{.+}} @__hipRegisterVar(ptr %[[HANDLE]], ptr @[[HOSTVAR]], ptr @[[VARNAME]]
;
; CHECK: #[[ATTR]] = {
; CHECK-SAME: kit_fb kit_tt(4)

target triple = "x86_64-unknown-linux-gnu"

@v137 = external global i32, align 4

define void @f(ptr %c, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp4.not = icmp eq i64 %n, 0
  br i1 %cmp4.not, label %forall.sync, label %forall.detach

forall.detach:
  %i.05 = phi i64 [ %inc, %forall.inc ], [ 0, %entry ]
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:
  %0 = load i32, ptr @v137, align 4
  %arrayidx = getelementptr inbounds i32, ptr %c, i64 %i.05
  store i32 %0, ptr %arrayidx, align 4
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

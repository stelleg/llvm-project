; Check that a launch call and a fat binary are present in the host. Check
; that the launch arguments are as expected.
;
; RUN: opt --tapir=cuda -passes='tapir-lowering<O2>' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-DAG: @[[FB:.+]] = constant {{.+}} #[[FBATTR:[0-9]+]]
; CHECK-DAG: @[[G_KNAME:.+]] = private unnamed_addr constant [{{[0-9]+}} x i8] c"[[KNAME:.+]]\00"
; CHECK-DAG: @[[G_KERNEL_PROPS:.+]] = private unnamed_addr constant {{.+}} zeroinitializer #[[KPATTR:[0-9]+]]
;
; CHECK: define {{.+}} @f(ptr {{.*}}%[[C:.+]], i64 {{.*}}%[[N:.+]])
;
; Create a stream
;
; CHECK: %[[STREAM:.+]] = {{.*}}call {{.+}} @llvm.kit.thread.stream(i32 2)
; CHECK-NOT: {{.*}}call {{.+}} @llvm.kit.async.prefetch.htod
;
; The actual launch. The arguments are:
;
;   - tapir target id
;   - fat binary global
;   - kernel name global string literal
;   - trip count
;   - threads per block (zero to indicate that it is unset)
;   - kernel properties global
;   - thread stream
;
; These are followed by a variable number of arguments that are to be passed to
; the kernel being launched. These are typically in the order
;
;   - start index
;   - trip count
;   - grain size
;   - ...
;
; where ... are the rest of the arguments to be passed to the kernel. Currently,
; we force the grain size to be 1. The start index is always 0.
;
; CHECK: %{{[0-9]+}} = {{.*}}call {{.+}} @llvm.kit.async.launch.kernel(
; CHECK-SAME: i32 2,
; CHECK-SAME: ptr {{.*}}@[[FB]],
; CHECK-SAME: ptr {{.*}}@[[G_KNAME]],
; CHECK-SAME: i64 %n,
; CHECK-SAME: i32 0,
; CHECK-SAME: ptr {{.*}}@[[G_KERNEL_PROPS]],
; CHECK-SAME: ptr %[[STREAM]],
; CHECK-SAME: i64 0,
; CHECK-SAME: i64 %n,
; CHECK-SAME: i64 1,
; CHECK-SAME: ptr %c,
; CHECK-SAME: i64 %n
; CHECK-SAME: )
;
; By default, we always enter a sync immediately after the launch. A later
; optimization pass may (re)move this if appropriate
; CHECK: call {{.+}} @llvm.kit.sync.stream(i32 2, ptr %[[STREAM]])
; CHECK: ret void
; CHECK-NEXT: }
;
; CHECK-DAG: #[[KPATTR]] = { kit_tt(2) "kit_kernel_props"="[[KNAME]]" }
; CHECK-DAG: #[[FBATTR]] = { kit_fb kit_tt(2) }

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

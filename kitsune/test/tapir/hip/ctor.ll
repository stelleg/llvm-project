; Check that the global ctor calls the appropriate functions in Kitsune's
; runtime depending on the command line arguments passed.
;
; RUN: opt --tapir=hip -S %s \
; RUN:     -passes='tapir-lowering<O2>,kit-ctors' \
; RUN:     --tapir-hip-arch=gfx906 \
; RUN:     --tapir-hip-sramecc=off \
; RUN:     --tapir-hip-xnack=on \
; RUN:     --tapir-hip-features="-sramecc,+xnack" \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     | FileCheck %s -check-prefix DEFAULT
;
; Currently, even if a max-threads-per-block option is not used, the max is set
; to 1024.
;
; DEFAULT: @[[FB:.+]] = constant [0 x i8] zeroinitializer
; DEFAULT-SAME: section ".hip_fatbin"
; DEFAULT-SAME: #[[FBATTR:[0-9]+]]
;
; DEFAULT: @[[BUNDLE:.+]] = internal constant {{.+}} { i32 1212764230, i32 1, ptr @[[FB]], ptr null }
; DEFAULT-SAME: section ".hipFatBinSegment"
;
; DEFAULT: @[[HANDLE:[.]kithip[.].+]] = internal global ptr null
;
; DEFAULT: @llvm.global_ctors = appending global
; DEFAULT-SAME: { i32 65536, ptr @[[CTOR:[.]kithip[.]ctor.*]], ptr null }
;
; FIXME: There is a bug where calling __kithip_destroy raises a segmentation
; fault or some other error which looks like memory corruption bug. As a
; temporary workaround, __kithip_destroy is not called, but it eventually
; should be once the issue is fixed.
;
; DEFAULT: define {{.*}} @[[DTOR:[.]kithip[.]dtor.*]]{{[ ]*}}(
; DEFAULT: call {{.+}} @__hipUnregisterFatBinary
; DEFAULT-NOT: call {{.+}} @llvm.kit.finalize(i32 4)
;
; DEFAULT: define {{.+}} @[[CTOR]]
; DEFAULT: call {{.+}} @llvm.kit.initialize(i32 4)
; DEFAULT: call {{.+}} @llvm.kit.enable.verbose(i8 0)
; DEFAULT: call {{.+}} @llvm.kit.enable.xnack(i8 1)
; DEFAULT: call {{.+}} @llvm.kit.enable.y.axis.launches(i32 4, i8 0)
; DEFAULT-NOT: call {{.+}} @llvm.kit.set.fixed.tpb(i32 4,
; DEFAULT: call {{.+}} @llvm.kit.set.max.tpb(i32 4, i32 1024)
; DEFAULT-DAG: %[[HC:.+]] = call {{.+}}__hipRegisterFatBinary(ptr @[[BUNDLE]])
; DEFAULT: store ptr %[[HC]], ptr @[[HANDLE]]
; DEFAULT: call {{.+}}atexit(ptr @[[DTOR]])
; DEFAULT: }
;
; DEFAULT: attributes #[[FBATTR]] = {
; DEFAULT-SAME: kit_fb kit_tt(4)
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=hip -S %s \
; RUN:     -passes='tapir-lowering<O2>,kit-ctors' \
; RUN:     --tapir-gpu-tpb=77 \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     | FileCheck %s -check-prefix TPB
;
; TPB-LABEL: kithip.ctor{{.*}}
; TPB: call {{.+}} @llvm.kit.set.fixed.tpb(i32 4, i32 77)
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=hip -S %s \
; RUN:     -passes='tapir-lowering<O2>,kit-ctors' \
; RUN:     --tapir-gpu-max-tpb=29 \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     | FileCheck %s -check-prefix MTPB
;
; MTPB-LABEL: kithip.ctor{{.*}}
; MTPB: call {{.+}} @llvm.kit.set.max.tpb(i32 4, i32 29)
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=hip -S %s \
; RUN:     -passes='tapir-lowering<O2>,kit-ctors' \
; RUN:     --tapir-verbose \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     | FileCheck %s -check-prefix VERBOSE
;
; RUN: opt --tapir=hip -S %s \
; RUN:     -passes='tapir-lowering<O2>,kit-ctors' \
; RUN:     --kitrt-verbose \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     | FileCheck %s -check-prefix VERBOSE
;
; VERBOSE-LABEL: kithip.ctor{{.*}}
; VERBOSE: call {{.+}} @llvm.kit.enable.verbose(i8 1)
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=hip -S %s \
; RUN:     -passes='tapir-lowering<O2>,kit-ctors' \
; RUN:     --tapir-hip-xnack=off \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     | FileCheck %s -check-prefix NOXNACK
;
; RUN: opt --tapir=hip -S %s \
; RUN:     -passes='tapir-lowering<O2>,kit-ctors' \
; RUN:     --tapir-hip-xnack=any \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     | FileCheck %s -check-prefix NOXNACK
;
; NOXNACK-LABEL: kithip.ctor{{.*}}
; NOXNACK: call {{.+}} @llvm.kit.enable.xnack(i8 0)
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=hip -S %s \
; RUN:     -passes='tapir-lowering<O2>,kit-ctors' \
; RUN:     -hipabi-y-launch \
; RUN:     --tapir-hip-runtime-bcs="%S/input/amd.bc" \
; RUN:     | FileCheck %s -check-prefix YLAUNCH
;
; YLAUNCH-LABEL: kithip.ctor{{.*}}
; YLAUNCH: call {{.+}} @llvm.kit.enable.y.axis.launches(i32 4, i8 1)
;
; ----------------------------------------------------------------------------

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
  %arrayidx = getelementptr inbounds i64, ptr %c, i64 %indvars.iv
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

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"llvm.loop.unroll.disable"}

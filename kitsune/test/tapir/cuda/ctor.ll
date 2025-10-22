; Check that the global ctor calls the appropriate functions in Kitsune's
; runtime depending on the command line arguments passed.
;
; RUN: opt --tapir=cuda --tapir-cuda-arch=sm_72 \
; RUN:     -passes='tapir-lowering<O2>,kit-ctors' \
; RUN:     -S %s \
; RUN:     | FileCheck %s -check-prefix DEFAULT
;
; Currently, even if a max-threads-per-block option is not used, the max is set
; to 1024.
;
; DEFAULT: @[[FB:.+]] = constant [0 x i8] zeroinitializer
; DEFAULT-SAME: section ".nv_fatbin"
; DEFAULT-SAME: #[[FBATTR:[0-9]+]]
;
; DEFAULT: @[[BUNDLE:.+]] = internal constant {{.+}} { i32 1180844977, i32 1, ptr @[[FB]], ptr null }
; DEFAULT-SAME: section ".nvFatBinSegment"
;
; DEFAULT: @[[HANDLE:[.]kitcuda[.].+]] = internal global ptr null
;
; DEFAULT: @llvm.global_ctors = appending global
; DEFAULT-SAME: { i32 65536, ptr @[[CTOR:[.]kitcuda[.]ctor.*]], ptr null }
;
; DEFAULT: define {{.*}} @[[DTOR:[.]kitcuda[.]dtor.*]]{{[ ]*}}(
; DEFAULT: %[[HD:.+]] = load ptr, ptr @[[HANDLE]]
; DEFAULT: call {{.+}} @__cudaUnregisterFatBinary(ptr %[[HD]])
; DEFAULT: call {{.+}} @llvm.kit.finalize(i32 2)
;
; DEFAULT: define {{.+}} @[[CTOR]]
; DEFAULT: call {{.+}} @llvm.kit.initialize(i32 2)
; DEFAULT: call {{.+}} @llvm.kit.enable.verbose(i8 0)
; DEFAULT-NOT: call {{.+}} @llvm.kit.set.fixed.tpb(i32 2,
; DEFAULT: call {{.+}} @llvm.kit.set.max.tpb(i32 2, i32 1024)
; DEFAULT-DAG: call {{.+}} @llvm.kit.enable.refine.launches(i32 2, i8 1)
; DEFAULT-DAG: %[[HC:.+]] = call {{.+}}__cudaRegisterFatBinary(ptr @[[BUNDLE]])
; DEFAULT: store ptr %[[HC]], ptr @[[HANDLE]]
; DEFAULT: call void @__cudaRegisterFatBinaryEnd(ptr %[[HC]])
; DEFAULT: call {{.+}}atexit(ptr @[[DTOR]])
; DEFAULT: }
;
; DEFAULT: #[[FBATTR]] = { kit_fb kit_tt(2) }
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda -passes='tapir-lowering<O2>,kit-ctors' \
; RUN:     -S %s --tapir-gpu-tpb=77 \
; RUN:     | FileCheck %s -check-prefix TPB
;
; TPB-LABEL: define {{.+}} @.kitcuda.ctor
; TPB: call {{.+}} @llvm.kit.set.fixed.tpb(i32 2, i32 77)
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda -passes='tapir-lowering<O2>,kit-ctors' \
; RUN:     -S %s --tapir-gpu-max-tpb=29 \
; RUN:     | FileCheck %s -check-prefix MTPB
;
; MTPB-LABEL: define {{.+}} @.kitcuda.ctor
; MTPB: call {{.+}} @llvm.kit.set.max.tpb(i32 2, i32 29)
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda -passes='tapir-lowering<O2>,kit-ctors' \
; RUN:     -S %s --tapir-verbose \
; RUN:     | FileCheck %s -check-prefix VERBOSE
;
; RUN: opt --tapir=cuda -passes='tapir-lowering<O2>,kit-ctors' \
; RUN:     -S %s --kitrt-verbose \
; RUN:     | FileCheck %s -check-prefix VERBOSE
;
; VERBOSE-LABEL: define {{.+}} @.kitcuda.ctor
; VERBOSE: call {{.+}} @llvm.kit.enable.verbose(i8 1)
;
; ----------------------------------------------------------------------------
;
; RUN: opt --tapir=cuda -passes='tapir-lowering<O2>,kit-ctors' \
; RUN:     -S %s -cuabi-refine-launches=false \
; RUN:     | FileCheck %s -check-prefix NOREFINE
;
; NOREFINE-LABEL: define {{.+}} @.kitcuda.ctor
; NOREFINE: call {{.+}} @llvm.kit.enable.refine.launches(i32 2, i8 0)
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
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"llvm.loop.unroll.disable"}

; Check that alloca instructions have the correct address space annotation
; added to them.
;
; RUN: opt --tapir=hip -passes='loop-spawning,emb-prepare' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: %[[A1:.+]] = alloca i64, align 8, addrspace(5)
; CHECK: %[[A2:.+]] = alloca i64, align 8, addrspace(5)
; CHECK: %[[CST1:.+]] = addrspacecast ptr addrspace(5) %[[A1]] to ptr
; CHECK: %[[CST2:.+]] = addrspacecast ptr addrspace(5) %[[A2]] to ptr
; CHECK: %[[V0:.+]] = ptrtoint ptr %[[CST1]] to i64
; CHECK: %[[V1:.+]] = ptrtoint ptr %[[CST2]] to i64

target triple = "x86_64-unknown-linux-gnu"

%"class.kitsune::mobile_ptr" = type { ptr addrspace(67) }

@stash = global ptr zeroinitializer

define void @_ZL20initialize_variablesiN7kitsune10mobile_ptrIfEES1_(i64 %nelr, ptr %variables, ptr %ff_variable) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp14 = icmp sgt i64 %nelr, 0
  br i1 %cmp14, label %preheader, label %forall.sync

preheader:
  %invariant.gep = getelementptr inbounds float, ptr %variables, i64 %nelr
  br label %forall.detach

forall.detach:
  %indvars.iv = phi i64 [ 0, %preheader ], [ %indvars.iv.next, %forall.inc ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  detach within %syncreg, label %forall.body, label %forall.inc

forall.body:
  %a1 = alloca i64, align 8
  %a2 = alloca i64, align 8
  %0 = ptrtoint ptr %a1 to i64
  %1 = ptrtoint ptr %a2 to i64
  %2 = add i64 %0, %1
  %3 = add i64 %2, %indvars.iv
  %4 = load float, ptr %ff_variable, align 4
  %arrayidx.i = getelementptr inbounds float, ptr %variables, i64 %3
  store float %4, ptr %arrayidx.i, align 4
  reattach within %syncreg, label %forall.inc

forall.inc:
  %exitcond.not = icmp eq i64 %indvars.iv.next, %nelr
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

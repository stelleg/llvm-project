; Non-constant global variabes used in a kernel (or device) function must be
; in a specific address space. However, simply setting the address space is not
; sufficient since we need to ensure that the change in address space does not
; affect the instructions in the body. For instance, if the global is passed to
; a device function, there may be a type mismatch if the device function does
; not expect pointers in a specific address space. In order to deal with this,
; we cast away the address space in every use of the global.
;
; RUN: opt --tapir=hip %s --tapir-hip-features="+16-bit-insts" \
; RUN:     -passes='loop-spawning,emb-prepare' \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK: @[[GV:.+]] = {{.*}}addrspace(1) global [16 x i64]
;
; CHECK-LABEL: define {{.+}} @id(
; CHECK-SAME: ptr %[[ARG0:[^,]+]],
; CHECK-SAME: i64 %[[ARG1:[^)]+]])
; CHECK: getelementptr inbounds i64, ptr addrspacecast (ptr addrspace(1) @[[GV]] to ptr), i64 %[[ARG1]]
;
; CHECK-LABEL: define {{.+}} @__kithip_loop_f
; CHECK: %[[IV:.+]] = phi i64
; CHECK: %[[V0:.+]] = tail call fastcc i64 @id(ptr addrspacecast (ptr addrspace(1) @[[GV]] to ptr), i64 %[[IV]])
; CHECK: %[[V1:.+]] = ptrtoint ptr addrspacecast (ptr addrspace(1) @[[GV]] to ptr) to i64
; CHECK: %[[V2:.+]] = getelementptr inbounds i64, ptr addrspacecast (ptr addrspace(1) @[[GV]] to ptr), i64 %[[IV]]
; CHECK: %[[V3:.+]] = load i64, ptr addrspacecast (ptr addrspace(1) @[[GV]] to ptr)
; CHECK: ptrtoint ptr %[[V2]] to i64

target triple = "x86_64-pc-linux-gnu"

@v137 = external global [16 x i64]

define i64 @id(ptr %p, i64 %iv) {
entry:
  %0 = getelementptr inbounds i64, ptr @v137, i64 %iv
  %1 = load i64, ptr %0
  ret i64 %1
}

define void @f(ptr %a, ptr %b, ptr %c, i64 %n) {
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
  %0 = tail call i64 @id(ptr @v137, i64 %indvars.iv)
  %1 = ptrtoint ptr @v137 to i64
  %2 = getelementptr inbounds i64, ptr @v137, i64 %indvars.iv
  %3 = load i64, ptr @v137
  %4 = add i64 %0, %1
  %5 = ptrtoint ptr %2 to i64
  %6 = mul i64 %4, %5
  %7 = sub i64 %6, %3
  store i64 %7, ptr @v137
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

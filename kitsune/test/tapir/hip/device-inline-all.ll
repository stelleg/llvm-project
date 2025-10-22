; Check that the command line option to inline all device functions is handled
; correctly.
;
; ------------------------------------------------------------------------------
;
; RUN: opt --tapir=hip --tapir-hip-arch=gfx90a %s \
; RUN:     -passes='tapir-lowering<O1>,emb-prepare' \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s -check-prefixes ALL,DEFAULT
;
; ALL: define {{.+}} @sieve{{.+}} #[[ATTRS_SIEVE:[0-9]+]]
; ALL: define {{.+}} @id{{.+}} #[[ATTRS_ID:[0-9]+]]
;
; DEFAULT-DAG: attributes #[[ATTRS_SIEVE]] = { kit_device nounwind "
; DEFAULT-DAG: attributes #[[ATTRS_ID]] = { kit_device noinline nounwind "
;
; ------------------------------------------------------------------------------
;
; RUN: opt --tapir=hip --tapir-hip-arch=gfx90a %s \
; RUN:     -passes='tapir-lowering<O1>,emb-prepare' -emb-inline-all \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s -check-prefixes ALL,INLINE
;
; INLINE-DAG: attributes #[[ATTRS_SIEVE]] = { alwaysinline kit_device nounwind "
; INLINE-DAG: attributes #[[ATTRS_ID]] = { kit_device noinline nounwind "
;
; ------------------------------------------------------------------------------

target triple = "x86_64-pc-linux-gnu"

; Function Attrs: noinline
define i64 @id(i64 %n) #0 {
  ret i64 %n
}

define dso_local i64 @sieve(i64 %0) {
  %2 = alloca [256 x i8], align 16
  call void @llvm.lifetime.start.p0(i64 256, ptr nonnull %2) #1
  br label %5

3:
  %4 = icmp slt i64 %0, 4
  br i1 %4, label %10, label %16

5:
  %6 = phi i64 [ 0, %1 ], [ %8, %5 ]
  %7 = getelementptr inbounds [256 x i8], ptr %2, i64 0, i64 %6
  store i8 1, ptr %7, align 1
  %8 = add nuw nsw i64 %6, 1
  %9 = icmp eq i64 %8, 256
  br i1 %9, label %3, label %5, !llvm.loop !5

10:
  %11 = add nsw i64 %0, -1
  %12 = getelementptr inbounds [256 x i8], ptr %2, i64 0, i64 %11
  %13 = load i8, ptr %12, align 1
  %14 = trunc nuw i8 %13 to i1
  %15 = add nsw i64 %0, -2
  br i1 %14, label %39, label %34

16:
  %17 = phi i64 [ %31, %30 ], [ 2, %3 ]
  %18 = phi i64 [ %32, %30 ], [ 4, %3 ]
  %19 = getelementptr inbounds [256 x i8], ptr %2, i64 0, i64 %17
  %20 = load i8, ptr %19, align 1
  %21 = trunc nuw i8 %20 to i1
  %22 = and i64 %18, 4294967295
  %23 = icmp sle i64 %22, %0
  %24 = and i1 %23, %21
  br i1 %24, label %25, label %30

25:
  %26 = phi i64 [ %28, %25 ], [ %22, %16 ]
  %27 = getelementptr inbounds [256 x i8], ptr %2, i64 0, i64 %26
  store i8 0, ptr %27, align 1
  %28 = add nuw nsw i64 %26, %17
  %29 = icmp sgt i64 %28, %0
  br i1 %29, label %30, label %25, !llvm.loop !6

30:
  %31 = add nuw nsw i64 %17, 1
  %32 = mul nuw nsw i64 %31, %31
  %33 = icmp sgt i64 %32, %0
  br i1 %33, label %10, label %16, !llvm.loop !7

34:
  %35 = getelementptr inbounds [256 x i8], ptr %2, i64 0, i64 %15
  %36 = load i8, ptr %35, align 1
  %37 = zext nneg i8 %36 to i64
  %38 = add nsw i64 %37, %0
  br label %39

39:
  %40 = phi i64 [ %38, %34 ], [ %15, %10 ]
  call void @llvm.lifetime.end.p0(i64 256, ptr nonnull %2) #1
  ret i64 %40
}

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
  %.call1 = call i64 @sieve(i64 %n)
  %.call2 = call i64 @id(i64 %n)
  %.sum = add i64 %.call1, %.call2
  store i64 %.sum, ptr %arrayidx, align 4
  reattach within %syncreg, label %forall.inc

forall.inc:
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !llvm.loop !0

forall.sync:
  sync within %syncreg, label %forall.end

forall.end:
  ret void
}

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
declare token @llvm.syncregion.start() #1

attributes #0 = { noinline }

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"tapir.loop.target", i32 4}
!3 = !{!"llvm.loop.unroll.disable"}
!4 = !{!"llvm.loop.mustprogress"}
!5 = !{!5, !4, !3}
!6 = distinct !{!6, !4, !3}
!7 = distinct !{!7, !4, !3}

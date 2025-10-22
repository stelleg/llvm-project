; Check that debug information makes it to the kernel module. This checks some
; basic debug info nodes.
;
; RUN: opt --tapir=hip -passes='tapir-lowering<O2>' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @__kithip_loop_test.cpp_6_3(
; CHECK-SAME: !dbg ![[DILOOP1:[0-9]+]]
; CHECK: getelementptr inbounds float{{.+}} !dbg ![[DILOC1:[0-9]+]]
; CHECK-LABEL: @__kithip_loop_test.cpp_11_3(
; CHECK-SAME: !dbg ![[DILOOP2:[0-9]+]]
; CHECK: getelementptr inbounds float{{.+}} !dbg ![[DILOC2:[0-9]+]]
; CHECK-LABEL: @__kithip_loop_test.cpp_14_3(
; CHECK-SAME: !dbg ![[DILOOP3:[0-9]+]]
; CHECK: getelementptr inbounds float{{.+}} !dbg ![[DILOC3:[0-9]+]]
;
; CHECK-COUNT-1: !DICompileUnit(language: DW_LANG_C_plus_plus
;
; CHECK: ![[DILOOP1]] = {{.*}}!DISubprogram(name: "scale",{{.*}} scope: ![[DIFILE:[0-9]+]],
; CHECK: ![[DIFILE]] = {{.*}}!DIFile(filename: "test.cpp",
; CHECK: ![[DIBODY1:[0-9]+]] = {{.*}}!DILexicalBlock(scope: ![[DILOOP1]], file: ![[DIFILE]], line: 6, column: 3)
; CHECK: ![[DIBLOCK1:[0-9]+]] = {{.*}}!DILexicalBlock(scope: ![[DIBODY1]], file: ![[DIFILE]], line: 6, column: 3)
; CHECK: ![[DILOC1]] = {{.*}}!DILocation(line: 7, column: 5, scope: ![[DIBLOCK1]])
;
; CHECK: ![[DILOOP2]] = {{.*}}!DISubprogram(name: "xlate",{{.*}} scope: ![[DIFILE]]
; CHECK: ![[DIBODY2:[0-9]+]] = {{.*}}!DILexicalBlock(scope: ![[DILOOP2]], file: ![[DIFILE]], line: 11, column: 3)
; CHECK: ![[DIBLOCK2:[0-9]+]] = {{.*}}!DILexicalBlock(scope: ![[DIBODY2]], file: ![[DIFILE]], line: 11, column: 3)
; CHECK: ![[DILOC2]] = {{.*}}!DILocation(line: 12, column: 5, scope: ![[DIBLOCK2]])
;
; CHECK: ![[DILOOP3]] = {{.*}}!DISubprogram(name: "xlate",{{.*}} scope: ![[DIFILE]]
; CHECK: ![[DIBODY3:[0-9]+]] = {{.*}}!DILexicalBlock(scope: ![[DILOOP3]], file: ![[DIFILE]], line: 14, column: 3)
; CHECK: ![[DIBLOCK3:[0-9]+]] = {{.*}}!DILexicalBlock(scope: ![[DIBODY3]], file: ![[DIFILE]], line: 14, column: 3)
; CHECK: ![[DILOC3]] = {{.*}}!DILocation(line: 15, column: 5, scope: ![[DIBLOCK3]])

target triple = "x86_64-unknown-linux-gnu"

define dso_local void @_Z5scalePffm(ptr %buf, float %factor, i64 %n) !dbg !261 {
entry:
  %syncreg = tail call token @llvm.syncregion.start(), !dbg !274
    #dbg_value(ptr %buf, !267, !DIExpression(), !274)
    #dbg_value(float %factor, !268, !DIExpression(), !274)
    #dbg_value(i64 %n, !269, !DIExpression(), !274)
    #dbg_value(i64 0, !270, !DIExpression(), !275)
  %cmp4.not = icmp eq i64 %n, 0, !dbg !276
  br i1 %cmp4.not, label %forall.sync, label %forall.detach, !dbg !277

forall.detach:
  %i.05 = phi i64 [ %inc, %forall.inc ], [ 0, %entry ]
    #dbg_value(i64 %i.05, !270, !DIExpression(), !275)
  detach within %syncreg, label %forall.body, label %forall.inc, !dbg !277

forall.body:
    #dbg_value(i64 %i.05, !272, !DIExpression(), !278)
  %arrayidx = getelementptr inbounds float, ptr %buf, i64 %i.05, !dbg !279
  %0 = load float, ptr %arrayidx, align 4, !dbg !280, !tbaa !281
  %mul = fmul float %factor, %0, !dbg !280
  store float %mul, ptr %arrayidx, align 4, !dbg !280, !tbaa !281
  reattach within %syncreg, label %forall.inc, !dbg !279

forall.inc:
  %inc = add nuw i64 %i.05, 1, !dbg !285
    #dbg_value(i64 %inc, !270, !DIExpression(), !275)
  %exitcond.not = icmp eq i64 %inc, %n, !dbg !276
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !dbg !277, !llvm.loop !286

forall.sync:
  sync within %syncreg, label %forall.end, !dbg !291

forall.end:
  ret void, !dbg !292
}

define void @_Z5xlatePffm(ptr %buf, float %dist, i64 %n) !dbg !293 {
entry:
  %syncreg2 = tail call token @llvm.syncregion.start(), !dbg !306
    #dbg_value(ptr %buf, !295, !DIExpression(), !306)
    #dbg_value(float %dist, !296, !DIExpression(), !306)
    #dbg_value(i64 %n, !297, !DIExpression(), !306)
    #dbg_value(i64 0, !298, !DIExpression(), !307)
  %cmp23.not = icmp eq i64 %n, 0, !dbg !308
  br i1 %cmp23.not, label %forall.sync, label %forall.detach, !dbg !309

forall.detach:
  %i.024 = phi i64 [ %inc, %forall.inc ], [ 0, %entry ]
    #dbg_value(i64 %i.024, !298, !DIExpression(), !307)
  detach within %syncreg2, label %forall.body, label %forall.inc, !dbg !309

forall.body:
    #dbg_value(i64 %i.024, !300, !DIExpression(), !310)
  %arrayidx = getelementptr inbounds float, ptr %buf, i64 %i.024, !dbg !311
  %0 = load float, ptr %arrayidx, align 4, !dbg !312, !tbaa !281
  %add = fadd float %dist, %0, !dbg !312
  store float %add, ptr %arrayidx, align 4, !dbg !312, !tbaa !281
  reattach within %syncreg2, label %forall.inc, !dbg !311

forall.inc:
  %inc = add nuw i64 %i.024, 1, !dbg !313
    #dbg_value(i64 %inc, !298, !DIExpression(), !307)
  %exitcond.not = icmp eq i64 %inc, %n, !dbg !308
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !dbg !309, !llvm.loop !314

forall.sync:
  sync within %syncreg2, label %forall.cond4.preheader, !dbg !316

forall.cond4.preheader:
    #dbg_value(i64 0, !302, !DIExpression(), !317)
  br i1 %cmp23.not, label %forall.sync14, label %forall.detach7, !dbg !318

forall.detach7:
  %i3.026 = phi i64 [ %inc13, %forall.inc12 ], [ 0, %forall.cond4.preheader ]
    #dbg_value(i64 %i3.026, !302, !DIExpression(), !317)
  detach within %syncreg2, label %forall.body8, label %forall.inc12, !dbg !318

forall.body8:
    #dbg_value(i64 %i3.026, !304, !DIExpression(), !319)
  %arrayidx10 = getelementptr inbounds float, ptr %buf, i64 %i3.026, !dbg !320
  %1 = load float, ptr %arrayidx10, align 4, !dbg !321, !tbaa !281
  %2 = tail call float @llvm.fmuladd.f32(float %dist, float 2.000000e+00, float %1), !dbg !321
  store float %2, ptr %arrayidx10, align 4, !dbg !321, !tbaa !281
  reattach within %syncreg2, label %forall.inc12, !dbg !320

forall.inc12:
  %inc13 = add nuw i64 %i3.026, 1, !dbg !322
    #dbg_value(i64 %inc13, !302, !DIExpression(), !317)
  %exitcond27.not = icmp eq i64 %inc13, %n, !dbg !323
  br i1 %exitcond27.not, label %forall.sync14, label %forall.detach7, !dbg !318, !llvm.loop !324

forall.sync14:
  sync within %syncreg2, label %forall.end15, !dbg !326

forall.end15:
  ret void, !dbg !327
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!253, !254, !255, !256, !257, !258, !259}
!llvm.ident = !{!260}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 20.1.2", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, imports: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "build", checksumkind: CSK_MD5, checksum: "d6203c6cc928ec16307d3c5f2aa122b6")
!2 = !{!3, !11, !15, !22, !26, !34, !39, !41, !50, !54, !58, !69, !71, !75, !79, !83, !88, !92, !96, !100, !104, !112, !116, !120, !122, !126, !130, !135, !141, !145, !149, !151, !159, !163, !171, !173, !177, !181, !185, !189, !194, !199, !204, !205, !206, !207, !209, !210, !211, !212, !213, !214, !215, !217, !218, !219, !220, !221, !222, !223, !224, !229, !230, !231, !232, !233, !234, !235, !236, !237, !238, !239, !240, !241, !242, !243, !244, !245, !246, !247, !248, !249, !250, !251, !252}
!3 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !5, file: !10, line: 52)
!4 = !DINamespace(name: "std", scope: null)
!5 = !DISubprogram(name: "abs", scope: !6, file: !6, line: 980, type: !7, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!6 = !DIFile(filename: "/usr/include/stdlib.h", directory: "", checksumkind: CSK_MD5, checksum: "775455349f6dd75df6b36d4b094321a4")
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIFile(filename: "/usr/lib/gcc/x86_64-pc-linux-gnu/13.3.1/../../../gcc/x86_64-pc-linux-gnu/13.3.1/include/c++/bits/std_abs.h", directory: "")
!11 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !12, file: !14, line: 131)
!12 = !DIDerivedType(tag: DW_TAG_typedef, name: "div_t", file: !6, line: 63, baseType: !13)
!13 = !DICompositeType(tag: DW_TAG_structure_type, file: !6, line: 59, size: 64, flags: DIFlagFwdDecl, identifier: "_ZTS5div_t")
!14 = !DIFile(filename: "/usr/lib/gcc/x86_64-pc-linux-gnu/13.3.1/../../../gcc/x86_64-pc-linux-gnu/13.3.1/include/c++/cstdlib", directory: "")
!15 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !16, file: !14, line: 132)
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "ldiv_t", file: !6, line: 71, baseType: !17)
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !6, line: 67, size: 128, flags: DIFlagTypePassByValue, elements: !18, identifier: "_ZTS6ldiv_t")
!18 = !{!19, !21}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !17, file: !6, line: 69, baseType: !20, size: 64)
!20 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !17, file: !6, line: 70, baseType: !20, size: 64, offset: 64)
!22 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !23, file: !14, line: 134)
!23 = !DISubprogram(name: "abort", scope: !6, file: !6, line: 730, type: !24, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: DISPFlagOptimized)
!24 = !DISubroutineType(types: !25)
!25 = !{null}
!26 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !27, file: !14, line: 136)
!27 = !DISubprogram(name: "aligned_alloc", scope: !6, file: !6, line: 724, type: !28, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!28 = !DISubroutineType(types: !29)
!29 = !{!30, !31, !31}
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!31 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !32, line: 18, baseType: !33)
!32 = !DIFile(filename: "lib/clang/20/include/__stddef_size_t.h", directory: "build", checksumkind: CSK_MD5, checksum: "2c44e821a2b1951cde2eb0fb2e656867")
!33 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!34 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !35, file: !14, line: 138)
!35 = !DISubprogram(name: "atexit", scope: !6, file: !6, line: 734, type: !36, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!36 = !DISubroutineType(types: !37)
!37 = !{!9, !38}
!38 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !24, size: 64)
!39 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !40, file: !14, line: 141)
!40 = !DISubprogram(name: "at_quick_exit", scope: !6, file: !6, line: 739, type: !36, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!41 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !42, file: !14, line: 144)
!42 = !DISubprogram(name: "atof", scope: !43, file: !43, line: 25, type: !44, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!43 = !DIFile(filename: "/usr/include/bits/stdlib-float.h", directory: "", checksumkind: CSK_MD5, checksum: "5b8ae17a9c8f951e8aefde76c3c6338d")
!44 = !DISubroutineType(types: !45)
!45 = !{!46, !47}
!46 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!47 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !48, size: 64)
!48 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !49)
!49 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!50 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !51, file: !14, line: 145)
!51 = !DISubprogram(name: "atoi", scope: !6, file: !6, line: 481, type: !52, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!52 = !DISubroutineType(types: !53)
!53 = !{!9, !47}
!54 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !55, file: !14, line: 146)
!55 = !DISubprogram(name: "atol", scope: !6, file: !6, line: 486, type: !56, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!56 = !DISubroutineType(types: !57)
!57 = !{!20, !47}
!58 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !59, file: !14, line: 147)
!59 = !DISubprogram(name: "bsearch", scope: !60, file: !60, line: 20, type: !61, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!60 = !DIFile(filename: "/usr/include/bits/stdlib-bsearch.h", directory: "", checksumkind: CSK_MD5, checksum: "f99fcd29986159d95c3009efc7923f1a")
!61 = !DISubroutineType(types: !62)
!62 = !{!30, !63, !63, !31, !31, !65}
!63 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !64, size: 64)
!64 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!65 = !DIDerivedType(tag: DW_TAG_typedef, name: "__compar_fn_t", file: !6, line: 948, baseType: !66)
!66 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !67, size: 64)
!67 = !DISubroutineType(types: !68)
!68 = !{!9, !63, !63}
!69 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !70, file: !14, line: 148)
!70 = !DISubprogram(name: "calloc", scope: !6, file: !6, line: 675, type: !28, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!71 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !72, file: !14, line: 149)
!72 = !DISubprogram(name: "div", scope: !6, file: !6, line: 992, type: !73, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!73 = !DISubroutineType(types: !74)
!74 = !{!12, !9, !9}
!75 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !76, file: !14, line: 150)
!76 = !DISubprogram(name: "exit", scope: !6, file: !6, line: 756, type: !77, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: DISPFlagOptimized)
!77 = !DISubroutineType(types: !78)
!78 = !{null, !9}
!79 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !80, file: !14, line: 151)
!80 = !DISubprogram(name: "free", scope: !6, file: !6, line: 687, type: !81, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!81 = !DISubroutineType(types: !82)
!82 = !{null, !30}
!83 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !84, file: !14, line: 152)
!84 = !DISubprogram(name: "getenv", scope: !6, file: !6, line: 773, type: !85, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!85 = !DISubroutineType(types: !86)
!86 = !{!87, !47}
!87 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !49, size: 64)
!88 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !89, file: !14, line: 153)
!89 = !DISubprogram(name: "labs", scope: !6, file: !6, line: 981, type: !90, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!90 = !DISubroutineType(types: !91)
!91 = !{!20, !20}
!92 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !93, file: !14, line: 154)
!93 = !DISubprogram(name: "ldiv", scope: !6, file: !6, line: 994, type: !94, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!94 = !DISubroutineType(types: !95)
!95 = !{!16, !20, !20}
!96 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !97, file: !14, line: 155)
!97 = !DISubprogram(name: "malloc", scope: !6, file: !6, line: 672, type: !98, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!98 = !DISubroutineType(types: !99)
!99 = !{!30, !31}
!100 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !101, file: !14, line: 157)
!101 = !DISubprogram(name: "mblen", scope: !6, file: !6, line: 1062, type: !102, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!102 = !DISubroutineType(types: !103)
!103 = !{!9, !47, !31}
!104 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !105, file: !14, line: 158)
!105 = !DISubprogram(name: "mbstowcs", scope: !6, file: !6, line: 1073, type: !106, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!106 = !DISubroutineType(types: !107)
!107 = !{!31, !108, !111, !31}
!108 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !109)
!109 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !110, size: 64)
!110 = !DIBasicType(name: "wchar_t", size: 32, encoding: DW_ATE_signed)
!111 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !47)
!112 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !113, file: !14, line: 159)
!113 = !DISubprogram(name: "mbtowc", scope: !6, file: !6, line: 1065, type: !114, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!114 = !DISubroutineType(types: !115)
!115 = !{!9, !108, !111, !31}
!116 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !117, file: !14, line: 161)
!117 = !DISubprogram(name: "qsort", scope: !6, file: !6, line: 970, type: !118, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!118 = !DISubroutineType(types: !119)
!119 = !{null, !30, !31, !31, !65}
!120 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !121, file: !14, line: 164)
!121 = !DISubprogram(name: "quick_exit", scope: !6, file: !6, line: 762, type: !77, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: DISPFlagOptimized)
!122 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !123, file: !14, line: 167)
!123 = !DISubprogram(name: "rand", scope: !6, file: !6, line: 573, type: !124, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!124 = !DISubroutineType(types: !125)
!125 = !{!9}
!126 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !127, file: !14, line: 168)
!127 = !DISubprogram(name: "realloc", scope: !6, file: !6, line: 683, type: !128, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!128 = !DISubroutineType(types: !129)
!129 = !{!30, !30, !31}
!130 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !131, file: !14, line: 169)
!131 = !DISubprogram(name: "srand", scope: !6, file: !6, line: 575, type: !132, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!132 = !DISubroutineType(types: !133)
!133 = !{null, !134}
!134 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!135 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !136, file: !14, line: 170)
!136 = !DISubprogram(name: "strtod", scope: !6, file: !6, line: 118, type: !137, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!137 = !DISubroutineType(types: !138)
!138 = !{!46, !111, !139}
!139 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !140)
!140 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !87, size: 64)
!141 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !142, file: !14, line: 171)
!142 = !DISubprogram(name: "strtol", linkageName: "__isoc23_strtol", scope: !6, file: !6, line: 215, type: !143, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!143 = !DISubroutineType(types: !144)
!144 = !{!20, !111, !139, !9}
!145 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !146, file: !14, line: 172)
!146 = !DISubprogram(name: "strtoul", linkageName: "__isoc23_strtoul", scope: !6, file: !6, line: 219, type: !147, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!147 = !DISubroutineType(types: !148)
!148 = !{!33, !111, !139, !9}
!149 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !150, file: !14, line: 173)
!150 = !DISubprogram(name: "system", scope: !6, file: !6, line: 923, type: !52, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!151 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !152, file: !14, line: 175)
!152 = !DISubprogram(name: "wcstombs", scope: !6, file: !6, line: 1077, type: !153, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!153 = !DISubroutineType(types: !154)
!154 = !{!31, !155, !156, !31}
!155 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !87)
!156 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !157)
!157 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !158, size: 64)
!158 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !110)
!159 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !160, file: !14, line: 176)
!160 = !DISubprogram(name: "wctomb", scope: !6, file: !6, line: 1069, type: !161, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!161 = !DISubroutineType(types: !162)
!162 = !{!9, !87, !110}
!163 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !164, entity: !165, file: !14, line: 204)
!164 = !DINamespace(name: "__gnu_cxx", scope: null)
!165 = !DIDerivedType(tag: DW_TAG_typedef, name: "lldiv_t", file: !6, line: 81, baseType: !166)
!166 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !6, line: 77, size: 128, flags: DIFlagTypePassByValue, elements: !167, identifier: "_ZTS7lldiv_t")
!167 = !{!168, !170}
!168 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !166, file: !6, line: 79, baseType: !169, size: 64)
!169 = !DIBasicType(name: "long long", size: 64, encoding: DW_ATE_signed)
!170 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !166, file: !6, line: 80, baseType: !169, size: 64, offset: 64)
!171 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !164, entity: !172, file: !14, line: 210)
!172 = !DISubprogram(name: "_Exit", scope: !6, file: !6, line: 768, type: !77, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: DISPFlagOptimized)
!173 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !164, entity: !174, file: !14, line: 214)
!174 = !DISubprogram(name: "llabs", scope: !6, file: !6, line: 984, type: !175, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!175 = !DISubroutineType(types: !176)
!176 = !{!169, !169}
!177 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !164, entity: !178, file: !14, line: 220)
!178 = !DISubprogram(name: "lldiv", scope: !6, file: !6, line: 998, type: !179, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!179 = !DISubroutineType(types: !180)
!180 = !{!165, !169, !169}
!181 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !164, entity: !182, file: !14, line: 231)
!182 = !DISubprogram(name: "atoll", scope: !6, file: !6, line: 493, type: !183, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!183 = !DISubroutineType(types: !184)
!184 = !{!169, !47}
!185 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !164, entity: !186, file: !14, line: 232)
!186 = !DISubprogram(name: "strtoll", linkageName: "__isoc23_strtoll", scope: !6, file: !6, line: 238, type: !187, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!187 = !DISubroutineType(types: !188)
!188 = !{!169, !111, !139, !9}
!189 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !164, entity: !190, file: !14, line: 233)
!190 = !DISubprogram(name: "strtoull", linkageName: "__isoc23_strtoull", scope: !6, file: !6, line: 243, type: !191, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!191 = !DISubroutineType(types: !192)
!192 = !{!193, !111, !139, !9}
!193 = !DIBasicType(name: "unsigned long long", size: 64, encoding: DW_ATE_unsigned)
!194 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !164, entity: !195, file: !14, line: 235)
!195 = !DISubprogram(name: "strtof", scope: !6, file: !6, line: 124, type: !196, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!196 = !DISubroutineType(types: !197)
!197 = !{!198, !111, !139}
!198 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!199 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !164, entity: !200, file: !14, line: 236)
!200 = !DISubprogram(name: "strtold", scope: !6, file: !6, line: 127, type: !201, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!201 = !DISubroutineType(types: !202)
!202 = !{!203, !111, !139}
!203 = !DIBasicType(name: "long double", size: 128, encoding: DW_ATE_float)
!204 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !165, file: !14, line: 244)
!205 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !172, file: !14, line: 246)
!206 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !174, file: !14, line: 248)
!207 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !208, file: !14, line: 249)
!208 = !DISubprogram(name: "div", linkageName: "_ZN9__gnu_cxx3divExx", scope: !164, file: !14, line: 217, type: !179, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!209 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !178, file: !14, line: 250)
!210 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !182, file: !14, line: 252)
!211 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !195, file: !14, line: 253)
!212 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !186, file: !14, line: 254)
!213 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !190, file: !14, line: 255)
!214 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !200, file: !14, line: 256)
!215 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !23, file: !216, line: 38)
!216 = !DIFile(filename: "/usr/lib/gcc/x86_64-pc-linux-gnu/13.3.1/../../../gcc/x86_64-pc-linux-gnu/13.3.1/include/c++/stdlib.h", directory: "", checksumkind: CSK_MD5, checksum: "3f24ff2a8eef595875da96e5466bd4aa")
!217 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !35, file: !216, line: 39)
!218 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !76, file: !216, line: 40)
!219 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !40, file: !216, line: 43)
!220 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !121, file: !216, line: 46)
!221 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !172, file: !216, line: 49)
!222 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !12, file: !216, line: 54)
!223 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !16, file: !216, line: 55)
!224 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !225, file: !216, line: 57)
!225 = !DISubprogram(name: "abs", linkageName: "_ZSt3absg", scope: !4, file: !10, line: 137, type: !226, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!226 = !DISubroutineType(types: !227)
!227 = !{!228, !228}
!228 = !DIBasicType(name: "__float128", size: 128, encoding: DW_ATE_float)
!229 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !42, file: !216, line: 58)
!230 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !51, file: !216, line: 59)
!231 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !55, file: !216, line: 60)
!232 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !59, file: !216, line: 61)
!233 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !70, file: !216, line: 62)
!234 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !208, file: !216, line: 63)
!235 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !80, file: !216, line: 64)
!236 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !84, file: !216, line: 65)
!237 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !89, file: !216, line: 66)
!238 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !93, file: !216, line: 67)
!239 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !97, file: !216, line: 68)
!240 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !101, file: !216, line: 70)
!241 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !105, file: !216, line: 71)
!242 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !113, file: !216, line: 72)
!243 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !117, file: !216, line: 74)
!244 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !123, file: !216, line: 75)
!245 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !127, file: !216, line: 76)
!246 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !131, file: !216, line: 77)
!247 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !136, file: !216, line: 78)
!248 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !142, file: !216, line: 79)
!249 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !146, file: !216, line: 80)
!250 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !150, file: !216, line: 81)
!251 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !152, file: !216, line: 83)
!252 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !160, file: !216, line: 84)
!253 = !{i32 7, !"Dwarf Version", i32 5}
!254 = !{i32 2, !"Debug Info Version", i32 3}
!255 = !{i32 1, !"wchar_size", i32 4}
!256 = !{i32 8, !"PIC Level", i32 2}
!257 = !{i32 7, !"PIE Level", i32 2}
!258 = !{i32 7, !"uwtable", i32 2}
!259 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!260 = !{!"clang version 20.1.2"}
!261 = distinct !DISubprogram(name: "scale", linkageName: "_Z5scalePffm", scope: !262, file: !262, line: 5, type: !263, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !266)
!262 = !DIFile(filename: "test.cpp", directory: "", checksumkind: CSK_MD5, checksum: "d6203c6cc928ec16307d3c5f2aa122b6")
!263 = !DISubroutineType(types: !264)
!264 = !{null, !265, !198, !31}
!265 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !198, size: 64)
!266 = !{!267, !268, !269, !270, !272}
!267 = !DILocalVariable(name: "buf", arg: 1, scope: !261, file: !262, line: 5, type: !265)
!268 = !DILocalVariable(name: "factor", arg: 2, scope: !261, file: !262, line: 5, type: !198)
!269 = !DILocalVariable(name: "n", arg: 3, scope: !261, file: !262, line: 5, type: !31)
!270 = !DILocalVariable(name: "i", scope: !271, file: !262, line: 6, type: !31)
!271 = distinct !DILexicalBlock(scope: !261, file: !262, line: 6, column: 3)
!272 = !DILocalVariable(name: "i", scope: !273, file: !262, line: 6, type: !31)
!273 = distinct !DILexicalBlock(scope: !271, file: !262, line: 6, column: 3)
!274 = !DILocation(line: 0, scope: !261)
!275 = !DILocation(line: 0, scope: !271)
!276 = !DILocation(line: 6, column: 27, scope: !273)
!277 = !DILocation(line: 6, column: 3, scope: !271)
!278 = !DILocation(line: 0, scope: !273)
!279 = !DILocation(line: 7, column: 5, scope: !273)
!280 = !DILocation(line: 7, column: 12, scope: !273)
!281 = !{!282, !282, i64 0}
!282 = !{!"float", !283, i64 0}
!283 = !{!"omnipotent char", !284, i64 0}
!284 = !{!"Simple C++ TBAA"}
!285 = !DILocation(line: 6, column: 32, scope: !273)
!286 = distinct !{!286, !277, !287, !288, !328, !290}
!287 = !DILocation(line: 7, column: 15, scope: !271)
!288 = !{!"tapir.loop.spawn.strategy", i32 1}
!290 = !{!"llvm.loop.unroll.disable"}
!291 = !DILocation(line: 6, column: 3, scope: !273)
!292 = !DILocation(line: 8, column: 1, scope: !261)
!293 = distinct !DISubprogram(name: "xlate", linkageName: "_Z5xlatePffm", scope: !262, file: !262, line: 10, type: !263, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !294)
!294 = !{!295, !296, !297, !298, !300, !302, !304}
!295 = !DILocalVariable(name: "buf", arg: 1, scope: !293, file: !262, line: 10, type: !265)
!296 = !DILocalVariable(name: "dist", arg: 2, scope: !293, file: !262, line: 10, type: !198)
!297 = !DILocalVariable(name: "n", arg: 3, scope: !293, file: !262, line: 10, type: !31)
!298 = !DILocalVariable(name: "i", scope: !299, file: !262, line: 11, type: !31)
!299 = distinct !DILexicalBlock(scope: !293, file: !262, line: 11, column: 3)
!300 = !DILocalVariable(name: "i", scope: !301, file: !262, line: 11, type: !31)
!301 = distinct !DILexicalBlock(scope: !299, file: !262, line: 11, column: 3)
!302 = !DILocalVariable(name: "i", scope: !303, file: !262, line: 14, type: !31)
!303 = distinct !DILexicalBlock(scope: !293, file: !262, line: 14, column: 3)
!304 = !DILocalVariable(name: "i", scope: !305, file: !262, line: 14, type: !31)
!305 = distinct !DILexicalBlock(scope: !303, file: !262, line: 14, column: 3)
!306 = !DILocation(line: 0, scope: !293)
!307 = !DILocation(line: 0, scope: !299)
!308 = !DILocation(line: 11, column: 27, scope: !301)
!309 = !DILocation(line: 11, column: 3, scope: !299)
!310 = !DILocation(line: 0, scope: !301)
!311 = !DILocation(line: 12, column: 5, scope: !301)
!312 = !DILocation(line: 12, column: 12, scope: !301)
!313 = !DILocation(line: 11, column: 32, scope: !301)
!314 = distinct !{!314, !309, !315, !288, !328, !290}
!315 = !DILocation(line: 12, column: 15, scope: !299)
!316 = !DILocation(line: 11, column: 3, scope: !301)
!317 = !DILocation(line: 0, scope: !303)
!318 = !DILocation(line: 14, column: 3, scope: !303)
!319 = !DILocation(line: 0, scope: !305)
!320 = !DILocation(line: 15, column: 5, scope: !305)
!321 = !DILocation(line: 15, column: 12, scope: !305)
!322 = !DILocation(line: 14, column: 32, scope: !305)
!323 = !DILocation(line: 14, column: 27, scope: !305)
!324 = distinct !{!324, !318, !325, !288, !328, !290}
!325 = !DILocation(line: 15, column: 22, scope: !303)
!326 = !DILocation(line: 14, column: 3, scope: !305)
!327 = !DILocation(line: 16, column: 1, scope: !293)
!328 = !{!"tapir.loop.target", i32 4}

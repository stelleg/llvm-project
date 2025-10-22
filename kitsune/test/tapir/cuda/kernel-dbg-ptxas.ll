; At the time of writing, ptxas does not support "optimized debugging". This is
; when -O<N> and -g are both provided to ptxas and where <N> is greater than 0.
; If one does provide, say -O2 -g to ptxas, it will fail with an error. However,
; --tapir=cuda requires optimizations in order to generate correct code. To work
; around this, if the backend finds debug information in the code, it
; automatically sets the ptxas optimization level to 0, unless the ptxas
; optimization level has been explicitly set by the user. Overriding the ptxas
; optimization level will result in a compiler crash.
;
; RUN: opt --tapir=cuda --tapir-cuda-arch=sm_80 \
; RUN:     --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:     -passes='kit-lowering<O2>,kit-cgfb' -cgfb-### %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s --check-prefixes ALL,DEFAULT
;
; RUN: not --crash opt --tapir=cuda --tapir-cuda-arch=sm_80 --cgfb-ptxas-O3 \
; RUN:     --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:     -passes='kit-lowering<O2>,kit-cgfb' -cgfb-### %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s --check-prefixes ALL,OVERRIDE
;
; ALL: /ptxas
; DEFAULT-SAME: --opt-level 0
; OVERRIDE-SAME: --opt-level 3

target triple = "x86_64-unknown-linux-gnu"

define void @_Z4add1Pfl(ptr %a, i64 %n) !dbg !261 {
entry:
  %syncreg = tail call token @llvm.syncregion.start(), !dbg !273
    #dbg_value(ptr %a, !267, !DIExpression(), !273)
    #dbg_value(i64 %n, !268, !DIExpression(), !273)
    #dbg_value(i64 0, !269, !DIExpression(), !274)
  %cmp4 = icmp sgt i64 %n, 0, !dbg !275
  br i1 %cmp4, label %forall.detach, label %forall.sync, !dbg !276

forall.detach:                                    ; preds = %entry, %forall.inc
  %i.05 = phi i64 [ %inc, %forall.inc ], [ 0, %entry ]
    #dbg_value(i64 %i.05, !269, !DIExpression(), !274)
  detach within %syncreg, label %forall.body, label %forall.inc, !dbg !276

forall.body:                                      ; preds = %forall.detach
    #dbg_value(i64 %i.05, !271, !DIExpression(), !277)
  %arrayidx = getelementptr inbounds nuw float, ptr %a, i64 %i.05, !dbg !278
  %0 = load float, ptr %arrayidx, align 4, !dbg !279, !tbaa !280
  %add = fadd float %0, 1.000000e+00, !dbg !279
  store float %add, ptr %arrayidx, align 4, !dbg !279, !tbaa !280
  reattach within %syncreg, label %forall.inc, !dbg !278

forall.inc:                                       ; preds = %forall.body, %forall.detach
  %inc = add nuw nsw i64 %i.05, 1, !dbg !284
    #dbg_value(i64 %inc, !269, !DIExpression(), !274)
  %exitcond.not = icmp eq i64 %inc, %n, !dbg !275
  br i1 %exitcond.not, label %forall.sync, label %forall.detach, !dbg !276, !llvm.loop !285

forall.sync:                                      ; preds = %forall.inc, %entry
  sync within %syncreg, label %forall.end, !dbg !289

forall.end:                                       ; preds = %forall.sync
  ret void, !dbg !290
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!253, !254, !255, !256, !257, !258, !259}
!llvm.ident = !{!260}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 20.1.2", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, imports: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "build", checksumkind: CSK_MD5, checksum: "8135dd26baf5d1dbd9d2effee149a28a")
!2 = !{!3, !11, !15, !22, !26, !34, !39, !41, !50, !54, !58, !69, !71, !75, !79, !83, !88, !92, !96, !100, !104, !112, !116, !120, !122, !126, !130, !135, !141, !145, !149, !151, !159, !163, !171, !173, !177, !181, !185, !189, !194, !199, !204, !205, !206, !207, !209, !210, !211, !212, !213, !214, !215, !217, !218, !219, !220, !221, !222, !223, !224, !229, !230, !231, !232, !233, !234, !235, !236, !237, !238, !239, !240, !241, !242, !243, !244, !245, !246, !247, !248, !249, !250, !251, !252}
!3 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !5, file: !10, line: 58)
!4 = !DINamespace(name: "std", scope: null)
!5 = !DISubprogram(name: "abs", scope: !6, file: !6, line: 837, type: !7, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!6 = !DIFile(filename: "/usr/include/stdlib.h", directory: "", checksumkind: CSK_MD5, checksum: "d0b67d4c866748c04ac2b355c26c1c70")
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIFile(filename: "/usr/lib/gcc/x86_64-pc-linux-gnu/15/include/g++-v15/bits/std_abs.h", directory: "", checksumkind: CSK_MD5, checksum: "e447352e9df05640e24a5f9f85d288ce")
!11 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !12, file: !14, line: 137)
!12 = !DIDerivedType(tag: DW_TAG_typedef, name: "div_t", file: !6, line: 62, baseType: !13)
!13 = !DICompositeType(tag: DW_TAG_structure_type, file: !6, line: 58, size: 64, flags: DIFlagFwdDecl, identifier: "_ZTS5div_t")
!14 = !DIFile(filename: "/usr/lib/gcc/x86_64-pc-linux-gnu/15/include/g++-v15/cstdlib", directory: "", checksumkind: CSK_MD5, checksum: "745c77d592b579358a91081122d152be")
!15 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !16, file: !14, line: 138)
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "ldiv_t", file: !6, line: 70, baseType: !17)
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !6, line: 66, size: 128, flags: DIFlagTypePassByValue, elements: !18, identifier: "_ZTS6ldiv_t")
!18 = !{!19, !21}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !17, file: !6, line: 68, baseType: !20, size: 64)
!20 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !17, file: !6, line: 69, baseType: !20, size: 64, offset: 64)
!22 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !23, file: !14, line: 140)
!23 = !DISubprogram(name: "abort", scope: !6, file: !6, line: 588, type: !24, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: DISPFlagOptimized)
!24 = !DISubroutineType(types: !25)
!25 = !{null}
!26 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !27, file: !14, line: 142)
!27 = !DISubprogram(name: "aligned_alloc", scope: !6, file: !6, line: 583, type: !28, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!28 = !DISubroutineType(types: !29)
!29 = !{!30, !31, !31}
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!31 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !32, line: 18, baseType: !33)
!32 = !DIFile(filename: "lib/clang/20/include/__stddef_size_t.h", directory: "build", checksumkind: CSK_MD5, checksum: "2c44e821a2b1951cde2eb0fb2e656867")
!33 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!34 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !35, file: !14, line: 144)
!35 = !DISubprogram(name: "atexit", scope: !6, file: !6, line: 592, type: !36, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!36 = !DISubroutineType(types: !37)
!37 = !{!9, !38}
!38 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !24, size: 64)
!39 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !40, file: !14, line: 147)
!40 = !DISubprogram(name: "at_quick_exit", scope: !6, file: !6, line: 597, type: !36, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!41 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !42, file: !14, line: 150)
!42 = !DISubprogram(name: "atof", scope: !43, file: !43, line: 25, type: !44, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!43 = !DIFile(filename: "/usr/include/bits/stdlib-float.h", directory: "", checksumkind: CSK_MD5, checksum: "ce60958b260b171e83db3307f1d644f0")
!44 = !DISubroutineType(types: !45)
!45 = !{!46, !47}
!46 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!47 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !48, size: 64)
!48 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !49)
!49 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!50 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !51, file: !14, line: 151)
!51 = !DISubprogram(name: "atoi", scope: !6, file: !6, line: 361, type: !52, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!52 = !DISubroutineType(types: !53)
!53 = !{!9, !47}
!54 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !55, file: !14, line: 152)
!55 = !DISubprogram(name: "atol", scope: !6, file: !6, line: 366, type: !56, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!56 = !DISubroutineType(types: !57)
!57 = !{!20, !47}
!58 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !59, file: !14, line: 153)
!59 = !DISubprogram(name: "bsearch", scope: !60, file: !60, line: 20, type: !61, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!60 = !DIFile(filename: "/usr/include/bits/stdlib-bsearch.h", directory: "", checksumkind: CSK_MD5, checksum: "1a798a38b25adee7bb680abce9ef568a")
!61 = !DISubroutineType(types: !62)
!62 = !{!30, !63, !63, !31, !31, !65}
!63 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !64, size: 64)
!64 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!65 = !DIDerivedType(tag: DW_TAG_typedef, name: "__compar_fn_t", file: !6, line: 805, baseType: !66)
!66 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !67, size: 64)
!67 = !DISubroutineType(types: !68)
!68 = !{!9, !63, !63}
!69 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !70, file: !14, line: 154)
!70 = !DISubprogram(name: "calloc", scope: !6, file: !6, line: 541, type: !28, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!71 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !72, file: !14, line: 155)
!72 = !DISubprogram(name: "div", scope: !6, file: !6, line: 849, type: !73, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!73 = !DISubroutineType(types: !74)
!74 = !{!12, !9, !9}
!75 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !76, file: !14, line: 156)
!76 = !DISubprogram(name: "exit", scope: !6, file: !6, line: 614, type: !77, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: DISPFlagOptimized)
!77 = !DISubroutineType(types: !78)
!78 = !{null, !9}
!79 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !80, file: !14, line: 157)
!80 = !DISubprogram(name: "free", scope: !6, file: !6, line: 563, type: !81, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!81 = !DISubroutineType(types: !82)
!82 = !{null, !30}
!83 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !84, file: !14, line: 158)
!84 = !DISubprogram(name: "getenv", scope: !6, file: !6, line: 631, type: !85, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!85 = !DISubroutineType(types: !86)
!86 = !{!87, !47}
!87 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !49, size: 64)
!88 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !89, file: !14, line: 159)
!89 = !DISubprogram(name: "labs", scope: !6, file: !6, line: 838, type: !90, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!90 = !DISubroutineType(types: !91)
!91 = !{!20, !20}
!92 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !93, file: !14, line: 160)
!93 = !DISubprogram(name: "ldiv", scope: !6, file: !6, line: 851, type: !94, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!94 = !DISubroutineType(types: !95)
!95 = !{!16, !20, !20}
!96 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !97, file: !14, line: 161)
!97 = !DISubprogram(name: "malloc", scope: !6, file: !6, line: 539, type: !98, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!98 = !DISubroutineType(types: !99)
!99 = !{!30, !31}
!100 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !101, file: !14, line: 163)
!101 = !DISubprogram(name: "mblen", scope: !6, file: !6, line: 919, type: !102, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!102 = !DISubroutineType(types: !103)
!103 = !{!9, !47, !31}
!104 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !105, file: !14, line: 164)
!105 = !DISubprogram(name: "mbstowcs", scope: !6, file: !6, line: 930, type: !106, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!106 = !DISubroutineType(types: !107)
!107 = !{!31, !108, !111, !31}
!108 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !109)
!109 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !110, size: 64)
!110 = !DIBasicType(name: "wchar_t", size: 32, encoding: DW_ATE_signed)
!111 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !47)
!112 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !113, file: !14, line: 165)
!113 = !DISubprogram(name: "mbtowc", scope: !6, file: !6, line: 922, type: !114, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!114 = !DISubroutineType(types: !115)
!115 = !{!9, !108, !111, !31}
!116 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !117, file: !14, line: 167)
!117 = !DISubprogram(name: "qsort", scope: !6, file: !6, line: 827, type: !118, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!118 = !DISubroutineType(types: !119)
!119 = !{null, !30, !31, !31, !65}
!120 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !121, file: !14, line: 170)
!121 = !DISubprogram(name: "quick_exit", scope: !6, file: !6, line: 620, type: !77, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: DISPFlagOptimized)
!122 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !123, file: !14, line: 173)
!123 = !DISubprogram(name: "rand", scope: !6, file: !6, line: 453, type: !124, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!124 = !DISubroutineType(types: !125)
!125 = !{!9}
!126 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !127, file: !14, line: 174)
!127 = !DISubprogram(name: "realloc", scope: !6, file: !6, line: 549, type: !128, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!128 = !DISubroutineType(types: !129)
!129 = !{!30, !30, !31}
!130 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !131, file: !14, line: 175)
!131 = !DISubprogram(name: "srand", scope: !6, file: !6, line: 455, type: !132, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!132 = !DISubroutineType(types: !133)
!133 = !{null, !134}
!134 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!135 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !136, file: !14, line: 176)
!136 = !DISubprogram(name: "strtod", scope: !6, file: !6, line: 117, type: !137, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!137 = !DISubroutineType(types: !138)
!138 = !{!46, !111, !139}
!139 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !140)
!140 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !87, size: 64)
!141 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !142, file: !14, line: 177)
!142 = !DISubprogram(name: "strtol", scope: !6, file: !6, line: 176, type: !143, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!143 = !DISubroutineType(types: !144)
!144 = !{!20, !111, !139, !9}
!145 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !146, file: !14, line: 178)
!146 = !DISubprogram(name: "strtoul", scope: !6, file: !6, line: 180, type: !147, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!147 = !DISubroutineType(types: !148)
!148 = !{!33, !111, !139, !9}
!149 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !150, file: !14, line: 179)
!150 = !DISubprogram(name: "system", scope: !6, file: !6, line: 781, type: !52, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!151 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !152, file: !14, line: 181)
!152 = !DISubprogram(name: "wcstombs", scope: !6, file: !6, line: 933, type: !153, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!153 = !DISubroutineType(types: !154)
!154 = !{!31, !155, !156, !31}
!155 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !87)
!156 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !157)
!157 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !158, size: 64)
!158 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !110)
!159 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !160, file: !14, line: 182)
!160 = !DISubprogram(name: "wctomb", scope: !6, file: !6, line: 926, type: !161, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!161 = !DISubroutineType(types: !162)
!162 = !{!9, !87, !110}
!163 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !164, entity: !165, file: !14, line: 210)
!164 = !DINamespace(name: "__gnu_cxx", scope: null)
!165 = !DIDerivedType(tag: DW_TAG_typedef, name: "lldiv_t", file: !6, line: 80, baseType: !166)
!166 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !6, line: 76, size: 128, flags: DIFlagTypePassByValue, elements: !167, identifier: "_ZTS7lldiv_t")
!167 = !{!168, !170}
!168 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !166, file: !6, line: 78, baseType: !169, size: 64)
!169 = !DIBasicType(name: "long long", size: 64, encoding: DW_ATE_signed)
!170 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !166, file: !6, line: 79, baseType: !169, size: 64, offset: 64)
!171 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !164, entity: !172, file: !14, line: 216)
!172 = !DISubprogram(name: "_Exit", scope: !6, file: !6, line: 626, type: !77, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: DISPFlagOptimized)
!173 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !164, entity: !174, file: !14, line: 222)
!174 = !DISubprogram(name: "llabs", scope: !6, file: !6, line: 841, type: !175, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!175 = !DISubroutineType(types: !176)
!176 = !{!169, !169}
!177 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !164, entity: !178, file: !14, line: 228)
!178 = !DISubprogram(name: "lldiv", scope: !6, file: !6, line: 855, type: !179, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!179 = !DISubroutineType(types: !180)
!180 = !{!165, !169, !169}
!181 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !164, entity: !182, file: !14, line: 240)
!182 = !DISubprogram(name: "atoll", scope: !6, file: !6, line: 373, type: !183, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!183 = !DISubroutineType(types: !184)
!184 = !{!169, !47}
!185 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !164, entity: !186, file: !14, line: 241)
!186 = !DISubprogram(name: "strtoll", scope: !6, file: !6, line: 200, type: !187, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!187 = !DISubroutineType(types: !188)
!188 = !{!169, !111, !139, !9}
!189 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !164, entity: !190, file: !14, line: 242)
!190 = !DISubprogram(name: "strtoull", scope: !6, file: !6, line: 205, type: !191, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!191 = !DISubroutineType(types: !192)
!192 = !{!193, !111, !139, !9}
!193 = !DIBasicType(name: "unsigned long long", size: 64, encoding: DW_ATE_unsigned)
!194 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !164, entity: !195, file: !14, line: 244)
!195 = !DISubprogram(name: "strtof", scope: !6, file: !6, line: 123, type: !196, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!196 = !DISubroutineType(types: !197)
!197 = !{!198, !111, !139}
!198 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!199 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !164, entity: !200, file: !14, line: 245)
!200 = !DISubprogram(name: "strtold", scope: !6, file: !6, line: 126, type: !201, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!201 = !DISubroutineType(types: !202)
!202 = !{!203, !111, !139}
!203 = !DIBasicType(name: "long double", size: 128, encoding: DW_ATE_float)
!204 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !165, file: !14, line: 253)
!205 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !172, file: !14, line: 255)
!206 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !174, file: !14, line: 257)
!207 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !208, file: !14, line: 258)
!208 = !DISubprogram(name: "div", linkageName: "_ZN9__gnu_cxx3divExx", scope: !164, file: !14, line: 225, type: !179, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!209 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !178, file: !14, line: 259)
!210 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !182, file: !14, line: 261)
!211 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !195, file: !14, line: 262)
!212 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !186, file: !14, line: 263)
!213 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !190, file: !14, line: 264)
!214 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !200, file: !14, line: 265)
!215 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !23, file: !216, line: 38)
!216 = !DIFile(filename: "/usr/lib/gcc/x86_64-pc-linux-gnu/15/include/g++-v15/stdlib.h", directory: "", checksumkind: CSK_MD5, checksum: "6b5a21b1805b4429608f31a862826533")
!217 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !35, file: !216, line: 39)
!218 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !76, file: !216, line: 40)
!219 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !40, file: !216, line: 43)
!220 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !121, file: !216, line: 46)
!221 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !172, file: !216, line: 49)
!222 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !12, file: !216, line: 54)
!223 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !16, file: !216, line: 55)
!224 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !225, file: !216, line: 57)
!225 = !DISubprogram(name: "abs", linkageName: "_ZSt3absg", scope: !4, file: !10, line: 143, type: !226, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
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
!261 = distinct !DISubprogram(name: "add1", linkageName: "_Z4add1Pfl", scope: !262, file: !262, line: 3, type: !263, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !266)
!262 = !DIFile(filename: "test.cpp", directory: "", checksumkind: CSK_MD5, checksum: "8135dd26baf5d1dbd9d2effee149a28a")
!263 = !DISubroutineType(types: !264)
!264 = !{null, !265, !20}
!265 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !198, size: 64)
!266 = !{!267, !268, !269, !271}
!267 = !DILocalVariable(name: "a", arg: 1, scope: !261, file: !262, line: 3, type: !265)
!268 = !DILocalVariable(name: "n", arg: 2, scope: !261, file: !262, line: 3, type: !20)
!269 = !DILocalVariable(name: "i", scope: !270, file: !262, line: 4, type: !20)
!270 = distinct !DILexicalBlock(scope: !261, file: !262, line: 4, column: 5)
!271 = !DILocalVariable(name: "i", scope: !272, file: !262, line: 4, type: !20)
!272 = distinct !DILexicalBlock(scope: !270, file: !262, line: 4, column: 5)
!273 = !DILocation(line: 0, scope: !261)
!274 = !DILocation(line: 0, scope: !270)
!275 = !DILocation(line: 4, column: 26, scope: !272)
!276 = !DILocation(line: 4, column: 5, scope: !270)
!277 = !DILocation(line: 0, scope: !272)
!278 = !DILocation(line: 5, column: 7, scope: !272)
!279 = !DILocation(line: 5, column: 12, scope: !272)
!280 = !{!281, !281, i64 0}
!281 = !{!"float", !282, i64 0}
!282 = !{!"omnipotent char", !283, i64 0}
!283 = !{!"Simple C++ TBAA"}
!284 = !DILocation(line: 4, column: 31, scope: !272)
!285 = distinct !{!285, !276, !286, !287, !291, !288}
!286 = !DILocation(line: 5, column: 15, scope: !270)
!287 = !{!"tapir.loop.spawn.strategy", i32 1}
!288 = !{!"llvm.loop.unroll.disable"}
!289 = !DILocation(line: 4, column: 5, scope: !272)
!290 = !DILocation(line: 6, column: 1, scope: !261)
!291 = !{!"tapir.loop.target", i32 2}

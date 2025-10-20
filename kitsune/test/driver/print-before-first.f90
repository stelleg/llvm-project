! REQUIRES: kitfc
!
! Check that the --print-before-first option is handled correctly.
!
! ----------------------------------------------------------------------------
! The --print-before-first option must be used with a Kitsune frontend
!
! RUN: not %flang -O1 --print-before-first %s \
! RUN:     -S -emit-llvm -o /dev/null 2>&1 \
! RUN:     | FileCheck %s -check-prefix FRONTEND
!
! FRONTEND: option '--print-before-first' must be used with a Kitsune frontend
!
! ----------------------------------------------------------------------------
! The --print-before-first option requires the --tapir option
!
! RUN: not %kitfc -O1 --print-before-first %s \
! RUN:     -S -emit-llvm -o /dev/null 2>&1 \
! RUN:     | FileCheck %s -check-prefix TT -allow-empty
!
! TT: --tapir is required with '--print-before-first'
!
! ----------------------------------------------------------------------------
! If the --print-before-first option has been implemented correctly, @_QQmain
! be called from @main. Since @_QQmain is empty, it will be absent after
! optimizations are run (because the empty body will have been inlined)
!
! RUN: %kitfc -O1 --tapir=serial --print-before-first %s \
! RUN:     -S -emit-llvm -o /dev/null 2>&1 \
! RUN:     | FileCheck %s -check-prefix EARLY
!
! EARLY: define {{.*}}i32 @main
! EARLY: call {{.+}} @_FortranAProgramStart
! EARLY: call {{.+}} @_QQmain
! EARLY: call {{.+}} @_FortranAProgramEndStatement
!
! ----------------------------------------------------------------------------

end program

// Check that the --print-before-first option is handled correctly.
//
// ----------------------------------------------------------------------------
// The --print-before-first option must be used with a Kitsune frontend
//
// RUN: not %clang -O1 --print-before-first %s \
// RUN:     -S -emit-llvm -o /dev/null 2>&1 \
// RUN:     | FileCheck %s -check-prefix FRONTEND
//
// FRONTEND: option '--print-before-first' must be used with a Kitsune frontend
//
// ----------------------------------------------------------------------------
// The --print-before-first option requires the --tapir option
//
// RUN: not %kitcc -O1 --print-before-first %s \
// RUN:     -S -emit-llvm -o /dev/null 2>&1 \
// RUN:     | FileCheck %s -check-prefix TT -allow-empty
//
// TT: --tapir is required with '--print-before-first'
//
// ----------------------------------------------------------------------------
// If the --print-before-first option has been implemented correctly, mem2reg
// will not have run, so a stack slot will have been created for the function
// argument.
//
// RUN: %kitcc -O1 --tapir=serial --print-before-first %s \
// RUN:     -S -emit-llvm -o /dev/null 2>&1 \
// RUN:     | FileCheck %s -check-prefix EARLY
//
// EARLY: define {{.+}}ptr @f(ptr {{.*}}%[[P:[^)]+]])
// EARLY: %[[SLOT:.+]] = alloca ptr
// EARLY: store ptr %[[P]], ptr %[[SLOT]]
// EARLY: %[[RV:.+]] = load ptr, ptr %[[SLOT]]
// EARLY: ret ptr %[[RV]]
//
// ----------------------------------------------------------------------------

void* f(void* p) {
  return p;
}

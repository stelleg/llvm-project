// RUN: %kitxx --tapir=serial -S -emit-llvm -O1 -o - %s \
// RUN:     | FileCheck %s

extern "C" {

double f1(double *[[kitsune::mobile]] ptr) { return *ptr; }

// CHECK-LABEL: @f1
// CHECK: %[[v:.+]] = load double, ptr addrspace(67) %ptr
// CHECK: ret double %[[v]]

char f2(char *[[kitsune::mobile]] ptr, int i) { return ptr[i]; }

// CHECK-LABEL: @f2
// CHECK: %[[gep:.+]] = getelementptr{{.+}} i8, ptr addrspace(67) %ptr, i64 {{.+}}
// CHECK: %[[v:.+]] = load i8, ptr addrspace(67) %[[gep]], {{.+}}
// CHECK: ret i8 %[[v]]

float *[[kitsune::mobile]] f3(float *[[kitsune::mobile]] ptr, int i) {
  return &ptr[i];
}

// CHECK-LABEL: @f3
// CHECK: %[[gep:.+]] = getelementptr{{.+}} float, ptr addrspace(67) %ptr, i64 {{.+}}
// CHECK: ret ptr addrspace(67) %[[gep]]

} // extern "C"

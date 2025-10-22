// RUN: %kitxx --tapir=nolo -S -emit-llvm -O0 -o - %s \
// RUN:     | FileCheck %s

typedef struct {
  int n;
} S1;

extern "C" S1 *[[kitsune::mobile]] s1 = nullptr;

// CHECK: @s1 ={{.*}} global ptr addrspace(67) null

extern "C" void f1(int *[[kitsune::mobile]] ptr) {}

// CHECK: void @f1(ptr addrspace(67) {{.+}})

extern "C" void *[[kitsune::mobile]] f2(void *[[kitsune::mobile]] ptr) {
  return ptr;
}

// CHECK: ptr addrspace(67) @f2(ptr addrspace(67) {{.+}})

extern "C" void f3() { void *[[kitsune::mobile]] ptr = nullptr; }

// CHECK-LABEL: @f3
// CHECK: %[[local:.+]] = alloca ptr addrspace(67)
// CHECK: store ptr addrspace(67) null, ptr %[[local]]

// Calling a function with a mobile attribute on a parameter must match the
// attributes exactly.
extern "C" void f4(int *[[kitsune::mobile]] ptr) { f1(ptr); }

// CHECK-LABEL: @f4
// CHECK: call void @f1(ptr addrspace(67) {{.+}})

// Return types must also match
extern "C" void *[[kitsune::mobile]] f5() { return f2(nullptr); }

// CHECK-LABEL: @f5
// CHECK: %[[ret:.+]] = call ptr addrspace(67) @f2(ptr addrspace(67) {{.*}}null)
// CHECK: ret ptr addrspace(67) %[[ret]]

// Passing, or returning a mobile pointer to function which does not accept one
// requires a cast.
extern "C" char f6(char *ptr) { return *ptr; }

extern "C" char f7(char *[[kitsune::mobile]] ptr) { return f6((char *)ptr); }

// CHECK-LABEL: @f7
// CHECK: %[[cst:.+]] = addrspacecast ptr addrspace(67) {{.+}} to ptr
// CHECK: call {{.*}}i8 @f6(ptr {{.*}}%[[cst]])

extern "C" float *f8(char *[[kitsune::mobile]] ptr) { return (float *)ptr; }

// CHECK-LABEL: @f8
// CHECK: %[[cst:.+]] = addrspacecast ptr addrspace(67) {{.+}} to ptr
// CHECK: ret ptr %[[cst]]

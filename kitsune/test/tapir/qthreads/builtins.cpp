// RUN: %kitxx --tapir=qthreads -O2 -S -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefixes=CHECK,DECLARES

#include <kitsune.h>

using namespace kitsune;

void allocate(mobile_ptr<int> &buf, size_t n) { buf.alloc(n); }

// CHECK-LABEL: _Z8allocate
// CHECK: call {{.+}} @__kitrt_default_mem_alloc({{.+}})

void deallocate(mobile_ptr<int> &buf) { buf.free(); }

// CHECK-LABEL: _Z10deallocate
// CHECK: call {{.+}} @__kitrt_default_mem_free({{.+}})

// DECLARES-DAG: declare noalias ptr addrspace(67) @__kitrt_default_mem_alloc

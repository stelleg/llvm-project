// RUN: %kitxx -Xclang -verify --tapir=serial -O1 -fsyntax-only %s

#include <string>

struct S1 {
  int n;
};

int [[kitsune::mobile]] i;
// expected-error@-1 {{'kitsune::mobile' can only be used on a pointer type}}

S1 [[kitsune::mobile]] s1;
// expected-error@-1 {{'kitsune::mobile' can only be used on a pointer type}}

double [[kitsune::mobile]] dbl[3];
// expected-error@-1 {{'kitsune::mobile' can only be used on a pointer type}}

std::string [[kitsune::mobile]] s;
// expected-error@-1 {{'kitsune::mobile' can only be used on a pointer type}}

void (*fptr)(int) [[kitsune::mobile]];
// expected-error@-1 {{'kitsune::mobile' can only be used on a pointer type}}

int fn(int) [[kitsune::mobile]];
// expected-error@-1 {{'kitsune::mobile' can only be used on a pointer type}}

int *f1(int *[[kitsune::mobile]] ptr) {
  // expected-error@+1 {{static_cast cannot be used to cast away mobile}}
  return static_cast<int *>(ptr);
}

S1 *f2(S1 *[[kitsune::mobile]] ptr) {
  // expected-error@+1 {{dynamic_cast cannot be used to cast away mobile}}
  return dynamic_cast<S1 *>(ptr);
}

int *f3(int *[[kitsune::mobile]] ptr) {
  // expected-error@+1 {{reinterpret_cast cannot be used to cast away mobile}}
  return reinterpret_cast<int *>(ptr);
}

int *f4(int *[[kitsune::mobile]] ptr) {
  // expected-error@+1 {{const_cast cannot be used to cast away mobile}}
  return const_cast<int *>(ptr);
}

int f5(int *ptr) { return *ptr; }

int f6(int *[[kitsune::mobile]] ptr) {
  // expected-error@+1 {{no matching function for call to 'f5'}}
  return f5(ptr);
  // expected-note-re@-5 {{candidate function not viable: {{.*}} does not accept mobile pointer. Consider using __kitsune_unsafe_cast}}
}

int *f7(int *[[kitsune::mobile]] ptr) {
  // expected-error@+1 {{cannot initialize return object}}
  return ptr;
}

void f8(int *[[kitsune::mobile]] arg) {
  // expected-error@+1 {{cannot initialize a variable}}
  int *local = arg;
}

void f9() {
  int *[[kitsune::mobile]] ptr = nullptr;
  int *local = nullptr;

  // expected-error-re@+1 {{assigning to {{.+}} discards qualifiers}}
  local = ptr;
}

int *g10 = nullptr;
void f10(int *[[kitsune::mobile]] ptr) {
  // expected-error-re@+1 {{assigning to {{.+}} discards qualifiers}}
  g10 = ptr;
}

float *f11(float *[[kitsune::mobile]] ptr, int idx) {
  // expected-error@+1 {{cannot initialize return object}}
  return &ptr[idx];
}

void f12(float *[[kitsune::mobile]] ptr, int i) {
  // expected-error-re@+1 {{cannot initialize a variable {{.+}} with an rvalue {{.+}}}}
  float *local = &ptr[i];
}

void f13(float *[[kitsune::mobile]] ptr, int i) {
  float *local = nullptr;
  // expected-error-re@+1 {{assigning to {{.+}} from {{.+}} discards qualifiers}}
  local = &ptr[i];
}

// Comparing mobile and non-mobile pointers is not allowed in C++, but it is
// allowed in C.
void f14(float *[[kitsune::mobile]] ptr1, float *ptr2) {
  // expected-error@+1 {{comparison of distinct pointer types}}
  return ptr1 == ptr2;
}

// RUN: %kitcc -Xclang -verify -fsyntax-only --tapir=nolo %s
// expected-no-diagnostics

class C;

// Attributes are valid on global variables.
C *_readwrite cptr;

// Attributes are valid on function parameters.
void f3(int _writeonly *out) {
  int *in;
  *out = *in;
}

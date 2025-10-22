// RUN: %kitxx -Xclang -verify -fsyntax-only --tapir=nolo %s
// expected-no-diagnostics

#include <cstdio>
#include <cstdlib>

#include <kitsune.h>

int main(int argc, char *argv[]) {
  [[tapir::strategy("seq")]]
  forall(int i = 0; i < 1024; ++i) {}

  [[tapir::strategy("dac")]]
  forall(int i = 0; i < 1024; ++i) {}

  [[tapir::strategy("dac")]]
  forall(int i = 0; i < 1024; ++i) {}

  return 0;
}

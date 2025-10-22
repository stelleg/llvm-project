// RUN: %kitxx -Xclang -verify -fsyntax-only --tapir=nolo %s
// expected-no-diagnostics

#include <kitsune.h>

int main(int argc, char *argv[]) {
  [[tapir::target("nolo")]]
  forall(int i = 0; i < 1024; ++i) {}

  [[tapir::target("serial")]]
  forall(int i = 0; i < 1024; ++i) {}

  [[tapir::target("cuda")]]
  forall(int i = 0; i < 1024; ++i) {}

  [[tapir::target("hip")]]
  forall(int i = 0; i < 1024; ++i) {}

  [[tapir::target("opencilk")]]
  forall(int i = 0; i < 1024; ++i) {}

  [[tapir::target("openmp")]]
  forall(int i = 0; i < 1024; ++i) {}

  [[tapir::target("qthreads")]]
  forall(int i = 0; i < 1024; ++i) {}

  [[tapir::target("realm")]]
  forall(int i = 0; i < 1024; ++i) {}

  return 0;
}

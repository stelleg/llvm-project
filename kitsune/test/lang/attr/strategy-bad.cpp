// RUN: %kitxx -Xclang -verify -fsyntax-only --tapir=nolo %s

#include <kitsune.h>

int main(int argc, char *argv[]) {
  [[tapir::strategy("greedy")]] // expected-error {{unknown strategy}}
  forall(int i = 0; i < 1024; ++i) {}

  [[tapir::strategy(seq)]] // expected-error {{'tapir::strategy' attribute requires a string}}
  forall(int i = 0; i < 1024; ++i) {}

  [[tapir::strategy()]] // expected-error {{'tapir::strategy' attribute takes one argument}}
  forall(int i = 0; i < 1024; ++i) {}

  [[tapir::strategy("seq", "gpu")]] // expected-error {{'tapir::strategy' attribute takes one argument}}
  forall(int i = 0; i < 1024; ++i) {}

  [[tapir::strategy("gpu")]] // expected-error {{'tapir::strategy' attribute only applies to 'forall' statement}}
  spawn s {}

  [[tapir::strategy("dac")]] // expected-error {{'tapir::strategy' attribute only applies to 'forall' statement}}
  if (argc == 1) {
    forall(int i = 0; i < 1024; ++i) {}
  }

  return 0;
}

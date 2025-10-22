// RUN: %kitxx -Xclang -verify -fsyntax-only --tapir=nolo %s

#include <kitsune.h>

void loop() {
  int i;

  forall(; i < 10; i++) {
    // expected-error@8 {{forall statement must have an initialization expression}}
  }

  forall(int j = 0;; j++) {
    // expected-error@12 {{forall statement must have a condition expression}}
  }

  forall(int j = 0; j < 10;) {
    // expected-error@16 {{forall statement must have an increment expression}}
  }

  forall(i = 0; i < 10; i++) {
    // expected-error@20 {{initializer in a forall statement must be a variable declaration}}
  }

  forall(int i = 0, j = 0; i < 10; i++, j++) {
    // expected-error@24 {{initializer in a forall statement must declare exactly one variable}}
  }

  forall(int i = 0; i < 10; i++) {
    if (i == 4) {
      break; // expected-error {{forall body may not have a break statement}}
    }
  }

  // continue statements are allowed in a forall.
  forall(int i = 0; i < 10; i++) {
    if (i == 4) {
      continue; // expected-no-error {{forall body may not have a continue statement}}
    }
  }
}

// Check that the tapir::target attribute on Kokkos::parallel_for statements is
// handled correctly.
//
// RUN: %kitxx -Xclang -verify -fsyntax-only -fkokkos -fkokkos-no-init  \
// RUN:   --tapir=serial -O1 %sysroot %s

#include <Kokkos_Core.hpp>

void f(float *A, int N) {
  // clang-format off
  [[tapir::target("i860")]] // expected-error {{unknown tapir target}}
  Kokkos::parallel_for(1024, KOKKOS_LAMBDA(const int i) {
    A[i] = i;
  });

  [[tapir::target(serial)]] // expected-error {{'tapir::target' attribute requires a string}}
  Kokkos::parallel_for(1024, KOKKOS_LAMBDA(const int i) {
    A[i] = i;
  });

  [[tapir::target()]] // expected-error {{'tapir::target' attribute takes one argument}}
  Kokkos::parallel_for(1024, KOKKOS_LAMBDA(const int i) {
    A[i] = i;
  });

  [[tapir::target("serial","-03")]] // expected-error {{'tapir::target' attribute takes one argument}}
  Kokkos::parallel_for(1024, KOKKOS_LAMBDA(const int i) {
    A[i] = i;
  });

  [[tapir::target("serial")]] // expected-error {{tapir target attribute on unsupported statement}}
  if (N == 1) {
    Kokkos::parallel_for(1024, KOKKOS_LAMBDA(const int i) {
      A[i] = i;
    });
  }
  // clang-format on
}

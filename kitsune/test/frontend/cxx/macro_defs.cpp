// -----------------------------------------------------------------------------
// Kitsune defines the __kitsune__ macro to allow the compiler to be identified
// as Kitsune. It also provides __kitsune_tt__ to identify the primary tapir
// target that has been set during the compilation. Check that these are set
// correctly.
//
// -----------------------------------------------------------------------------
//
// RUN: %kitxx -E %s -o - \
// RUN:      | FileCheck --check-prefixes=ID,NOTAPIR %s
// RUN: %kitxx --tapir=serial -O1 -E %s -o - \
// RUN:      | FileCheck --check-prefixes=ID,TAPIR %s
//
// -----------------------------------------------------------------------------
// The compile ID macros are also defined when just using clang. We may want to
// change this so they are only set when using the Kitsune frontend.
//
// RUN: %clangxx -E %s -o - | FileCheck --check-prefixes=ID,NOTAPIR %s
//
// -----------------------------------------------------------------------------
//
// ID: return 1;
// ID-NOT: return 0;
// NOTAPIR: return "";
// TAPIR: return "serial";
//
// -----------------------------------------------------------------------------

int isKitsune() {
#ifdef __kitsune__
  return 1;
#else
  return 0;
#endif // __kitsune
}

const char *getTapirTarget() {
#ifdef __kitsune_tt__
  return __kitsune_tt__;
#else
  return "";
#endif // __kitsune_tt_
}

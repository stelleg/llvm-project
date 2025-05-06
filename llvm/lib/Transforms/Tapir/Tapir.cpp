//===- Tapir.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements common infrastructure for libLLVMTapirOpts.a, which
// implements several transformations over the Tapir/LLVM intermediate
// representation, including the C bindings for that library.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Transforms/Tapir.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Transforms/Tapir.h"
#include "llvm/Transforms/Tapir/TapirTargetIDs.h"

using namespace llvm;

namespace llvm {

raw_ostream &operator<<(raw_ostream &os, const TapirTargetID &Target) {
  switch (Target) {
  case TapirTargetID::None:
    return os << "none";
  case TapirTargetID::Serial:
    return os << "serial";
  case TapirTargetID::Cuda:
    return os << "cuda";
  case TapirTargetID::Hip:
    return os << "hip";
  case TapirTargetID::GPU:
    return os << "gpu";
  case TapirTargetID::Lambda:
    return os << "lambda";
  case TapirTargetID::OMPTask:
    return os << "omptask";
  case TapirTargetID::OpenCilk:
    return os << "opencilk";
  case TapirTargetID::OpenMP:
    return os << "openmp";
  case TapirTargetID::Qthreads:
    return os << "qthreads";
  case TapirTargetID::Realm:
    return os << "realm";
  case TapirTargetID::Last_TapirTargetID:
    return os << "<invalid>";
  }
  // We don't put this in a default block in the switch above because it results
  // in compiler warnings about default blocks in a switch where all enumeration
  // values are handled. But we want this error in case a new tapir target is
  // added, but this code is not updated.
  llvm_unreachable("Tapir target not handled");
}

} // namespace llvm

/// initializeTapirOpts - Initialize all passes linked into the
/// TapirOpts library.
void llvm::initializeTapirOpts(PassRegistry &Registry) {
  initializeLoopSpawningTIPass(Registry);
  initializeLowerTapirToTargetPass(Registry);
  initializeTaskCanonicalizePass(Registry);
  initializeTaskSimplifyPass(Registry);
  initializeDRFScopedNoAliasWrapperPassPass(Registry);
  initializeLoopStripMinePass(Registry);
  initializeSerializeSmallTasksPass(Registry);
}

//===- DomDagUpdater.h - DomDag/Post DomDag Updater ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the DomDagUpdater class, which provides a uniform way to
// update dominator tree related data structures.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DOMDAGUPDATER_H
#define LLVM_ANALYSIS_DOMDAGUPDATER_H

#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/GenericDomTree.h"
#include <functional>
#include <vector>

namespace llvm {
class DomDagUpdater {
public:
  explicit DomDagUpdater()  {}
  DomDagUpdater(DominatorTree &DT_)
      : DT(&DT_) {}

  /// Returns true if it holds a DominatorTree.
  bool hasDomTree() const { return DT != nullptr; }

  /// Notify DTU that the entry block was replaced.
  /// Recalculate all available trees and flush all BasicBlocks
  /// awaiting deletion immediately.
  SmallVector<std::pair<BasicBlock*, BasicBlock*>, 4> addDagEdges(Function &F);

private:

  DominatorTree *DT = nullptr;

};
} // namespace llvm

#endif // LLVM_ANALYSIS_DOMTREEUPDATER_H

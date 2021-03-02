//===- RefinePhi.cpp - Refine Phis with Dominator DAG Tool ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RefinePhi class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/RefinePhiWithDomDAG.h"
#include "llvm/Analysis/PhiValues.h"
#include "llvm/Analysis/DomDagUpdater.h"
#include "llvm/InitializePasses.h"
#include<iostream>

using namespace llvm;

#define DEBUG_TYPE "refinephi"

namespace {

// The legacy pass of RefinePhis.
struct RefinePhisLegacyPass : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid

  RefinePhisLegacyPass() : FunctionPass(ID) {
    initializeRefinePhisLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();

  }

  bool runOnFunction(Function &F) override;
};

} // end anonymous namespace

char RefinePhisLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(RefinePhisLegacyPass, "refine-phis",
                      "Refine phis with domdag", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(RefinePhisLegacyPass, "refine-phis",
                    "Refine phis with domdag", false, false)

// Create the legacy RefinePhisPass.
FunctionPass *llvm::createRefinePhisPass() {
  return new RefinePhisLegacyPass();
}

static bool refinePhis(Function &F, DominatorTree &DT) {
  auto ddu = DomDagUpdater(DT);  
  auto pv = PhiValues(F); 
  auto bbs = ddu.addDagEdges(F); 
  bool changed = false; 
  for(auto p : bbs){
    BasicBlock* dominator = p.first; 
    for(auto &i : *dominator){
      for(Use &U : i.uses()) {
        Instruction *user = cast<Instruction>(U.getUser());
        if(isa<PHINode>(user)){
          for(Use &PU : user->uses()){
            Instruction *puser = cast<Instruction>(PU.getUser());
            if(DT.dominates(&i, puser)){
              printf("found phi to replace!\n"); 
              PU.set(&i); 
              changed = true; 
            }
          }
        }
      }
    }
  }
      
  return changed; 
}

bool RefinePhisLegacyPass::runOnFunction(Function &F) {
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  return refinePhis(F, DT);
}

PreservedAnalyses RefinePhisPass::run(Function &F,
                                      FunctionAnalysisManager &AM) {
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  if (!refinePhis(F, DT))
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}

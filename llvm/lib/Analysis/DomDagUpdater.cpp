//===- DomDagUpdater.cpp - DomDag/Post DomDag Updater --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the DomDagUpdater class, which provides a uniform way
// to update dominator tree related data structures.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DomDagUpdater.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/GenericDomTree.h"
#include <algorithm>
#include <functional>
#include <utility>

namespace llvm {

SmallVector<std::pair<BasicBlock*, BasicBlock*>, 4>  
DomDagUpdater::addDagEdges(Function& F){
  // We record each branch variable and the basic block whos terminator it
  // controls. 
  DenseMap<Value*, SmallSet<BasicBlock*, 4>*> vbrs; 
  for(auto &BB : F){
    if(auto *br = dyn_cast<BranchInst>(BB.getTerminator())){
      if(br->isConditional() && br->getNumSuccessors() == 2){
        Value* v = br->getCondition(); 
        auto bbs = vbrs.find(v); 
        if(bbs != vbrs.end()){
          bbs->second->insert(&BB); 
        }
        else{
          SmallSet<BasicBlock*, 4>* bbs = new SmallSet<BasicBlock*, 4>(); 
          bbs->insert(&BB); 
          vbrs.insert({v, bbs}); 
        }
      }
    }
  }
  SmallVector<std::pair<BasicBlock*, BasicBlock*>, 4> res; 
  for(auto kv : vbrs){
    if(auto bbs = kv.second){
      for(auto bb = bbs->begin(); bb != bbs->end(); bb++){
        for(auto bbn = bb; ++bbn != bbs->end(); ){
          auto term = (*bb)->getTerminator(); 
          auto termn = (*bbn)->getTerminator(); 
          if(DT->dominates(*bb, *bbn)){
            auto bbl = term->getSuccessor(0); 
            auto bbln = termn->getSuccessor(0); 
            DT->insertEdge(bbl, bbln); 
            res.push_back({bbl,bbl}); 
            auto bbr = term->getSuccessor(1); 
            auto bbrn = termn->getSuccessor(1); 
            DT->insertEdge(bbr, bbrn); 
            res.push_back({bbr,bbrn}); 
          } else if(DT->dominates(*bbn, *bb)){
            auto bbl = term->getSuccessor(0); 
            auto bbln = termn->getSuccessor(0); 
            DT->insertEdge(bbln, bbl); 
            res.push_back({bbln,bbl}); 
            auto bbr = term->getSuccessor(1); 
            auto bbrn = termn->getSuccessor(1); 
            DT->insertEdge(bbrn, bbr); 
            res.push_back({bbrn,bbr}); 
          }
        }
      }
    }
  }
  return res; 
}

} // namespace llvm

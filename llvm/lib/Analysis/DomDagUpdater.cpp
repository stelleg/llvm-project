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
#include <stdio.h>
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
  printf("running dag edges\n"); 
  // We record each branch variable and the basic block whos terminator it
  // controls. 
  DenseMap<Value*, SmallSet<BasicBlock*, 4>*> vbrs; 
  for(BasicBlock &BB : F){
    if(auto *br = dyn_cast<BranchInst>(BB.getTerminator())){
      if(br->isConditional() && br->getNumSuccessors() == 2){
        Value* v = br->getCondition(); 
        auto bbs = vbrs.find(v); 
        if(bbs != vbrs.end()){
          printf("found common condition!\n");
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
  printf("done collecting conditions\n"); 
  SmallVector<std::pair<BasicBlock*, BasicBlock*>, 4> res; 
  for(auto kv : vbrs){
    auto bbs = kv.second;
    if(bbs->size() > 1){  
      printf("processing common condition\n"); 
      SmallVector<BasicBlock*> stack; 
      for(BasicBlock *bb : *bbs){
        for(BasicBlock *bbn : stack){
          printf("processing basic block pair\n"); 
          auto term = bb->getTerminator(); 
          auto termn = bbn->getTerminator(); 
          printf(" terminators ok\n"); 
          if(DT->dominates(bb, bbn)){
            printf(" dominates forward\n"); 
            auto bbl = term->getSuccessor(0); 
            auto bbln = termn->getSuccessor(0); 
            DT->addDominanceRelation(bbl, bbln); 
            res.push_back({bbl,bbl}); 
            auto bbr = term->getSuccessor(1); 
            auto bbrn = termn->getSuccessor(1); 
            DT->addDominanceRelation(bbr, bbrn); 
            res.push_back({bbr,bbrn}); 
            printf(" handled dominates forward\n"); 
          } else if(DT->dominates(bbn, bb)){
            printf(" dominates backwards\n"); 
            auto bbl = term->getSuccessor(0); 
            auto bbln = termn->getSuccessor(0); 
            printf(" got false successors\n"); 
            DT->addDominanceRelation(bbln, bbl); 
            printf(" added dom rel\n"); 
            res.push_back({bbln,bbl}); 
            auto bbr = term->getSuccessor(1); 
            auto bbrn = termn->getSuccessor(1); 
            printf(" got true successors\n"); 
            DT->addDominanceRelation(bbrn, bbr); 
            res.push_back({bbrn,bbr}); 
            printf(" handled dominates backwards\n"); 
          }
        }
        stack.push_back(bb); 
      }
    }
  }
  return res; 
}

} // namespace llvm

//===- GenericDomGrove.h - Generic dominator groves for graphs ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This implements dominator dag relations via sets of pairs of dominator trees
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_GENERICDOMGROVE_H
#define LLVM_SUPPORT_GENERICDOMGROVE_H

#include<llvm/Support/GenericDomTree.h>
#include<llvm/Support/Path.h>
#include<llvm/ADT/SmallSet.h>
#include<llvm/ADT/SmallPtrSet.h>
#include<llvm/Demangle/Demangle.h>
#include<llvm/IR/Instructions.h>
#include "llvm/IR/Constants.h"
#include<fstream>
#include<iostream>
#include<llvm/ADT/Statistic.h>

//#include<llvm/Transforms/Utils/BasicBlockUtils.h>

#define DEBUG_TYPE "domgrove"
STATISTIC(NumDomDag, "Number of context sensitive domination relations calls"); 
STATISTIC(NumDom, "Number of total domination relations calls"); 
STATISTIC(DomDagFailures, "Cases where tree is more precise than dag"); 
STATISTIC(NumSharedCond, "Number of cases of shared variables"); 

namespace llvm {
void ReplaceInstWithInst(Instruction *From, Instruction *To);


// Grove updates can either be standard cfg updates or updates of 
// condition variables, changing the grove dominance relations
template<typename NodePtr, typename CondPtr> 
class GroveUpdate : public cfg::Update<NodePtr> {
  CondPtr Cond = nullptr; 
  
  public:
    GroveUpdate(NodePtr n, CondPtr c) : Cond(c) {this->From = n;} ;
    bool isCfgUpdate() { Cond == nullptr; }
}; 

/// Core dominator grove base class.
///
/// This class is a generic template over graph nodes. It is instantiated for
/// various graphs in the LLVM IR or in the code generator.
template <typename NodeT, typename Cond, bool IsPostDom>
class DominatorGroveBase : public DominatorTreeBase<NodeT, IsPostDom> {

private:
  using DomTree = DominatorTreeBase<NodeT, IsPostDom>; 
  using Bunch = SmallVector<std::unique_ptr<DomTree>>; 
  using Grove = SmallVector<std::unique_ptr<Bunch>>; 
	Grove grove;

public:
  using ParentPtr = decltype(std::declval<NodeT*>()->getParent());
  using ParentType = std::remove_pointer_t<ParentPtr>;
  using CondUpdate = GroveUpdate<NodeT, Cond*>; 
  using UpdateType = cfg::Update<NodeT*>;
  using UpdateKind = cfg::UpdateKind;
  static constexpr UpdateKind Delete = UpdateKind::Delete;

  Grove computeGrove() const { 
    // Construct bunches
    // We add the constraint that NodeT must have a function that returns an
    // optional for a branch condition. For a basic block, this returns a
    // value pointer, which is null if there isn't one
    Grove bunches; 
    int found = 0; 
    DenseMap<const Cond*, SmallPtrSet<NodeT *, 2>> m;
    for(auto &dtn : this->DomTreeNodes){
      auto tn = dtn.get();
      auto bb = tn ->getBlock();
      if(const Cond* cv = bb->getCond()){
        if(!isa<UndefValue>(cv) && !isa<Constant>(cv)){
          auto s = m.find(cv);
          if(s != m.end()){
            // Found multiple basic blocks with the same condition variable!
            found++; 
            s->second.insert(bb);
          } else {
            SmallPtrSet<NodeT*, 2> sn; 
            sn.insert(bb); 
            m[cv] = sn;
          }
        }
      }
      // If any blocks don't have a terminator, we're in a broken state and fall back to 
      if(!bb->getTerminator()){
        return bunches;
      }
    }

    if(auto bb = dyn_cast<BasicBlock>(this->getRoot())){
      if(found > 0)
        LLVM_DEBUG(dbgs() << "found " << found << " basic blocks with shared condition variables in " << demangle(bb->getParent()->getName()) << "\n"); 
    }

    for(auto &c : m){
      auto s = c.second; 
      if(s.size() > 1){
        if(auto *cv = dyn_cast<Value>(c.first)){
          LLVM_DEBUG(dbgs() << "  shared condition variable: " << *cv << "\n");    
          NumSharedCond++; 
        }
        auto tt = DominatorTreeBase<NodeT, IsPostDom>::copy(); 
        auto tf = DominatorTreeBase<NodeT, IsPostDom>::copy();
				SmallVector<UpdateType> tu;
				SmallVector<UpdateType> fu; 

        // We avoid creating a copy of the full cfg by modifying the cfg then creating the dt
        // TODO: Ew, ad-hoc polymorphism
        // TODO: Batch updates
        //DenseMap<NodeT*, BranchInst*> oldBranches;
        for(auto *bb : s){
          if(auto *BB = dyn_cast<BasicBlock>(bb)){
            BranchInst* old = dyn_cast<BranchInst>(BB->getTerminator()); 
            if(old->getSuccessor(1) != old->getSuccessor(0)){
              tu.emplace_back(UpdateType(Delete, BB, old->getSuccessor(1))); 
              fu.emplace_back(UpdateType(Delete, BB, old->getSuccessor(0))); 
            }
          }
        }

				tt->applyPhantomUpdates(tu);
				tf->applyPhantomUpdates(fu); 

        auto p = std::make_unique<Bunch>();
        p->push_back(std::move(tt)); 
        p->push_back(std::move(tf));
        bunches.push_back(std::move(p)); 
      }
    }
    return bunches;
  }

	void update() {
    //grove.clear(); 
    //grove = computeGrove(); 
	}

  bool isReachableFromEntry(const NodeT *A) const {
    assert(!this->isPostDominator() &&
           "This is not implemented for post dominators");

    bool naive = DominatorTreeBase<NodeT, IsPostDom>::isReachableFromEntry(A);; 
    Grove lgrove = computeGrove(); 
    bool any = false;
    for(auto &b : lgrove){
      for(auto &t : *b){
        any |= t->isReachableFromEntry(A);
      }
    }
    return lgrove.size() > 0 ? any : naive; 
  }

  bool isReachableFromEntry(const DomTreeNodeBase<NodeT> *A) const { return A; }

  bool dominates(const DomTreeNodeBase<NodeT> *A,
                 const DomTreeNodeBase<NodeT> *B) const {
    NumDom++; 
    // A node trivially dominates itself.
    if (B == A)
      return true;

    // An unreachable node is dominated by anything.
    if (!isReachableFromEntry(B))
      return true;

    // And dominates nothing.
    if (!isReachableFromEntry(A))
      return false;

    Grove lgrove = computeGrove(); 

    // A node dominates another iff it dominates for every variant 
		bool anyDom = false;
    LLVM_DEBUG(this->print(dbgs())); 
    bool naive = DominatorTreeBase<NodeT, IsPostDom>::dominates(A,B);; 
    for(auto& b : lgrove){
      bool dom = true;
      bool istrue = true; 
      for(auto &t : *b){
        istrue = not(istrue); 
        LLVM_DEBUG(dbgs() << (istrue ? "true case:\n" : "false case:\n")); 
        LLVM_DEBUG(t->print(dbgs())); 
        LLVM_DEBUG(dbgs() << A->getBlock()->getName() << 
          " dominates " << B->getBlock()->getName() << " = " 
          << t->dominates(A->getBlock(),B->getBlock()) << "\n"); 
        dom &= t->dominates(A->getBlock(),B->getBlock());
      }
      anyDom |= dom; 
    }

    if(lgrove.size() > 0 && naive && !anyDom) DomDagFailures++;  

    if(!naive && anyDom){
      NumDomDag++; 
      LLVM_DEBUG(dbgs() << "Context sensitive domination: " << 
        A->getBlock()->getName() << " dominates " <<  
        B->getBlock()->getName() << "\n"); 
      /*
      std::string fname = llvm::sys::path::filename(
        A->getBlock()->getParent()->getParent()->getSourceFileName()).str(); 
      std::cout << fname << ": " << 
        A->getBlock()->getName().str() << " dominates " << 
        B->getBlock()->getName().str() << "\n"; 
      std::ofstream logfile;
      logfile.open("/tmp/domdag-" + fname, std::ios::out | std::ios::app); 
      logfile << A->getBlock()->getName().str() << " dominates " << 
                 B->getBlock()->getName().str() << "\n"; 
			*/
    }

    return naive || anyDom; 
  }

  bool dominates(const NodeT *A, const NodeT *B) const;

	void applyUpdates(ArrayRef<CondUpdate> Updates) {
		DominatorTreeBase<NodeT, IsPostDom>::applyUpdates(Updates); 
		update();
	}

	void applyUpdates(ArrayRef<UpdateType> Updates) {
		DominatorTreeBase<NodeT, IsPostDom>::applyUpdates(Updates); 
		update();
	}

  void applyUpdates(ArrayRef<UpdateType> Updates,
                    ArrayRef<UpdateType> PostViewUpdates) {
		DominatorTreeBase<NodeT, IsPostDom>::applyUpdates(Updates, PostViewUpdates);
		update();
  }

  void insertEdge(NodeT *From, NodeT *To) {
		DominatorTreeBase<NodeT, IsPostDom>::insertEdge(From, To);
		update(); 
  }

  void deleteEdge(NodeT *From, NodeT *To) {
		DominatorTreeBase<NodeT, IsPostDom>::deleteEdge(From, To);
		update();
	}

  void recalculate(ParentType &Func) {
		DominatorTreeBase<NodeT, IsPostDom>::recalculate(Func);
		update();
	}

  void recalculate(ParentType &Func, ArrayRef<UpdateType> Updates) {
		DominatorTreeBase<NodeT, IsPostDom>::recalculate(Func, Updates);
		update(); 
	}
}; 

template <typename NodeT, typename Cond, bool IsPostDom>
bool DominatorGroveBase<NodeT, Cond, IsPostDom>::dominates(const NodeT *A,
                                                    const NodeT *B) const {
  if (A == B)
    return true;

  // Cast away the const qualifiers here. This is ok since
  // this function doesn't actually return the values returned
  // from getNode.
  return dominates(this->getNode(const_cast<NodeT *>(A)),
                   this->getNode(const_cast<NodeT *>(B)));
}

}
#undef DEBUG_TYPE
#endif // LLVM_SUPPORT_GENERICDOMGROVE_H

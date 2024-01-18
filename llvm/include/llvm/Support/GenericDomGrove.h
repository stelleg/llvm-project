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
#include<llvm/ADT/SmallSet.h>
#include<llvm/ADT/SmallPtrSet.h>
#include<llvm/Demangle/Demangle.h>
#include<llvm/IR/Instructions.h>
//#include<llvm/Transforms/Utils/BasicBlockUtils.h>

#define DEBUG_TYPE "domgrove"

namespace llvm {
void ReplaceInstWithInst(Instruction *From, Instruction *To);

/// Core dominator grove base class.
///
/// This class is a generic template over graph nodes. It is instantiated for
/// various graphs in the LLVM IR or in the code generator.
template <typename NodeT, typename Cond, bool IsPostDom>
class DominatorGroveBase : public DominatorTreeBase<NodeT, IsPostDom> {

private:
	SmallVector<std::unique_ptr<SmallVector<std::unique_ptr<DominatorTreeBase<NodeT, IsPostDom>>,2>>,2> bunches;

public:
  using ParentPtr = decltype(std::declval<NodeT*>()->getParent());
  using ParentType = std::remove_pointer_t<ParentPtr>;
  using UpdateType = cfg::Update<NodeT*>;
  using UpdateKind = cfg::UpdateKind;
  static constexpr UpdateKind Insert = UpdateKind::Insert;
  static constexpr UpdateKind Delete = UpdateKind::Delete;

	void update() {
    // Construct bunches
    // We add the constraint that NodeT must have a function that returns an
    // optional for a branch condition. For a basic block, this returns a
    // value pointer, which is null if there isn't one
		bunches.clear();
    int found = 0; 
    DenseMap<const Cond*, SmallPtrSet<NodeT *, 2>> m;
    for(auto &dtn : this->DomTreeNodes){
      auto tn = dtn.second.get();
      auto bb = tn ->getBlock();
      if(const Cond* cv = bb->getCond()){
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

    if(auto bb = dyn_cast<BasicBlock>(this->getRoot())){
      if(found > 0)
        LLVM_DEBUG(dbgs() << "found " << found << " basic blocks with shared condition variables in " << demangle(bb->getParent()->getName()) << "\n"); 
    }

    for(auto &c : m){
      auto s = c.second; 
      if(s.size() > 1){
        if(auto *cv = dyn_cast<Value>(c.first)){
          LLVM_DEBUG(dbgs() << "  shared condition variable: " << cv->getName() << "\n");    
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
						/*
            auto oldCopy = BranchInst::Create(old->getSuccessor(0), old->getSuccessor(1), old->getCondition());  // We create a copy because replaceinst will delete this one
            oldBranches[BB] = oldCopy;  
            auto tb = BranchInst::Create(old->getSuccessor(0)); 
            ReplaceInstWithInst(old, tb); 
						*/	
						tu.emplace_back(UpdateType(Delete, BB, old->getSuccessor(1))); 
						fu.emplace_back(UpdateType(Delete, BB, old->getSuccessor(0))); 
          }
        }

/*
        for(auto *bb : s){
          if(auto *BB = dyn_cast<BasicBlock>(bb)){
            BranchInst* old = oldBranches[BB]; 
            auto oldCopy = BranchInst::Create(old->getSuccessor(0), old->getSuccessor(1), old->getCondition());  // We create a copy because replaceinst will delete this one
            oldBranches[BB] = oldCopy;  
            ReplaceInstWithInst(BB->getTerminator(), old); 
          }
        }

        for(auto *bb : s){
          if(auto *BB = dyn_cast<BasicBlock>(bb)){
            auto fb = BranchInst::Create(oldBranches[BB]->getSuccessor(1)); 
            auto *t = BB->getTerminator(); 
            ReplaceInstWithInst(t, fb); 
            tf->deleteEdge(bb, oldBranches[BB]->getSuccessor(0));
          }
        }

        for(auto *bb : s){
          if(auto *BB = dyn_cast<BasicBlock>(bb)){
            auto *t = BB->getTerminator();
            ReplaceInstWithInst(t, oldBranches[BB]); 
          }
        }
				*/

				tt->applyUpdates(tu);
				tf->applyUpdates(fu); 

        auto p = std::make_unique<SmallVector<std::unique_ptr<DominatorTreeBase<NodeT, IsPostDom> >, 2> >();
        p->push_back(std::move(tt)); 
        p->push_back(std::move(tf));
        bunches.push_back(std::move(p)); 
      }
    }
	}

  bool dominates(const DomTreeNodeBase<NodeT> *A,
                 const DomTreeNodeBase<NodeT> *B) const {
    // A node dominates another iff it dominates for every variant 
		bool anyDom = DominatorTreeBase<NodeT, IsPostDom>::dominates(A,B);
    LLVM_DEBUG(this->print(dbgs())); 
    bool naive = anyDom; 
    for(auto& b : bunches){
      bool dom = true;
      for(auto &t : *b){
        LLVM_DEBUG(t->print(dbgs())); 
        LLVM_DEBUG(dbgs() << A->getBlock()->getName() << 
          " dominates " << B->getBlock()->getName() << " = " 
          << t->dominates(A->getBlock(),B->getBlock()) << "\n"); 
        dom &= t->dominates(A->getBlock(),B->getBlock());
      }
      anyDom |= dom; 
    }
    if(!naive && anyDom) LLVM_DEBUG(dbgs() << "Found a context sensitive domination!\n"); 
    return anyDom; 
  }

  bool dominates(const NodeT *A, const NodeT *B) const;

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

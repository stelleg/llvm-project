//===- RealmABI.h - Interface to the Qthreads runtime ----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is a simple pass wrapper around the PromoteMemToReg function call
// exposed by the Utils library.
//
//===----------------------------------------------------------------------===//
#ifndef REALM_ABI_H_
#define REALM_ABI_H_

#define REALM_ENABLE_C_BINDINGS TRUE

//#include "llvm/Transforms/Tapir/LoopSpawning.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"

namespace llvm {

class RealmABI : public TapirTarget {

  Type *TaskFuncPtrTy = nullptr;

  //Opaque Realm RTS functions
  FunctionCallee RealmGetNumProcs = nullptr;
  FunctionCallee RealmSpawn = nullptr;
  FunctionCallee RealmSync = nullptr;
  FunctionCallee RealmInitRuntime = nullptr;

  //Accessors for opaque Realm RTS functions
  FunctionCallee get_realmGetNumProcs();
  FunctionCallee get_realmSpawn();
  FunctionCallee get_realmSync();
  FunctionCallee get_realmInitRuntime();

public:
  RealmABI(Module &M);
  ~RealmABI() {} // may need to add something to clear lists, etc.
  
  
  Value *GetOrCreateWorker8(Function &F) override final;
  void createSync(SyncInst &inst, ValueToValueMapTy &DetachCtxToStackFrame)
    override final;

  Function *createDetach(DetachInst &Detach,
                         ValueToValueMapTy &DetachCtxToStackFrame,
                         DominatorTree &DT, AssumptionCache &AC) override final;
  void preProcessFunction(Function &F) override final;
  void postProcessFunction(Function &F) override final;
  void postProcessHelper(Function &F) override final;
};

}  // end of llvm namespace

#endif

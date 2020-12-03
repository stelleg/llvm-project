//===- HIPABI.h - Interface to the Kitsune HIP back end ------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Kitsune HIP ABI to convert Tapir instructions to
// calls into the Kitsune runtime system for NVIDIA GPU code.
//
//===----------------------------------------------------------------------===//
#ifndef HIP_ABI_H_
#define HIP_ABI_H_

#include "llvm/Transforms/Tapir/LoweringUtils.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"

namespace llvm {

class DataLayout;
class TargetMachine;
class HIPABI;

class AMDGCNLoop : public LoopOutlineProcessor {
  friend class HIPABI;

protected:
  static unsigned NextKernelID;
  unsigned MyKernelID;
  Module AMDGCNM;
  TargetMachine *AMDGCNTargetMachine;
  GlobalVariable *AMDGCNGlobal;

  FunctionCallee GetThreadIdx = nullptr;
  FunctionCallee GetBlockIdx = nullptr;
  FunctionCallee GetBlockDim = nullptr;

  void EmitAMDGCN(raw_pwrite_stream *OS);
  void makeFatBinaryString();
  virtual Function *makeModuleCtorFunction() { return nullptr; }
  virtual Function *makeModuleDtorFunction() { return nullptr; }
public:
  AMDGCNLoop(Module &M);

  void setupLoopOutlineArgs(
      Function &F, ValueSet &HelperArgs, SmallVectorImpl<Value *> &HelperInputs,
      ValueSet &InputSet, const SmallVectorImpl<Value *> &LCArgs,
      const SmallVectorImpl<Value *> &LCInputs,
      const ValueSet &TLInputsFixed)
    override final;
  unsigned getIVArgIndex(const Function &F, const ValueSet &Args) const
    override final;
  unsigned getLimitArgIndex(const Function &F, const ValueSet &Args) const
    override final;
  void postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                          ValueToValueMapTy &VMap) override final;
};

class HIPLoop : public AMDGCNLoop {
private:
  Type *Dim3Ty = nullptr;
  Type *HIPStreamTy = nullptr;

  FunctionCallee HIPLaunchKernel = nullptr;
  FunctionCallee HIPPopCallConfig = nullptr;
  FunctionCallee HIPPushCallConfig = nullptr;

  GlobalVariable *GpuBinaryHandle = nullptr;

  FunctionType *getRegisterGlobalsFnTy() const;
  FunctionType *getCallbackFnTy() const;
  FunctionType *getRegisterLinkedBinaryFnTy() const;

  Function *makeRegisterGlobalsFn();
  Function *makeModuleCtorFunction() override;
  Function *makeModuleDtorFunction() override;

  /// Keeps track of kernel launch stubs emitted in this module
  struct KernelInfo {
    Function *Kernel; // HostFunc
    StringRef DeviceFunc;
  };
  SmallVector<KernelInfo, 16> EmittedKernels;
public:
  HIPLoop(Module &M);

  void processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                               DominatorTree &DT) override final;
};

class KitsuneHIPLoop : public AMDGCNLoop {
private:
  FunctionCallee KitsuneHIPInit = nullptr;
  FunctionCallee KitsuneGPUInitKernel = nullptr;
  FunctionCallee KitsuneGPUInitField = nullptr;
  FunctionCallee KitsuneGPUSetRunSize = nullptr;
  FunctionCallee KitsuneGPURunKernel = nullptr;
  FunctionCallee KitsuneGPUFinish = nullptr;
public:
  KitsuneHIPLoop(Module &M);

  void processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                               DominatorTree &DT) override final;
};

class HIPABI : public TapirTarget {
  AMDGCNLoop *LOP = nullptr;
public:
  HIPABI(Module &M) : TapirTarget(M) {}
  ~HIPABI() {
    if (LOP)
      delete LOP;
  }
  Value *lowerGrainsizeCall(CallInst *GrainsizeCall) override final;
  void lowerSync(SyncInst &SI) override final;

  void addHelperAttributes(Function &F) override final {}
  void preProcessFunction(Function &F, TaskInfo &TI,
                          bool OutliningTapirLoops) override final;
  void postProcessFunction(Function &F, bool OutliningTapirLoops)
    override final;
  void postProcessHelper(Function &F) override final;

  void preProcessOutlinedTask(Function &F, Instruction *DetachPt,
                              Instruction *TaskFrameCreate,
                              bool IsSpawner) override final;
  void postProcessOutlinedTask(Function &F, Instruction *DetachPt,
                               Instruction *TaskFrameCreate,
                               bool IsSpawner) override final;
  void preProcessRootSpawner(Function &F) override final;
  void postProcessRootSpawner(Function &F) override final;
  void processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT)
    override final;

  LoopOutlineProcessor *getLoopOutlineProcessor(const TapirLoopInfo *TL)
    override final;
};
}

#endif

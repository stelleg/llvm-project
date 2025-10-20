//===- HipABI.h - Tapir target for Kitsune's hip runtime --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Copyright (c) 2021, 2023 Los Alamos National Security, LLC.
//  All rights reserved.
//
// Copyright 2021, 2023. Los Alamos National Security, LLC. This
//  software was produced under U.S. Government contract
//  DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which
//  is operated by Los Alamos National Security, LLC for the
//  U.S. Department of Energy. The U.S. Government has rights to use,
//  reproduce, and distribute this software.  NEITHER THE GOVERNMENT
//  NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS
//  OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.
//  If software is modified to produce derivative works, such modified
//  software should be clearly marked, so as not to confuse it with
//  the version available from LANL.
//
//  Additionally, redistribution and use in source and binary forms,
//  with or without modification, are permitted provided that the
//  following conditions are met:
//
// Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above
//      copyright notice, this list of conditions and the following
//      disclaimer in the documentation and/or other materials provided
//      with the distribution.
//
//    * Neither the name of Los Alamos National Security, LLC, Los
//      Alamos National Laboratory, LANL, the U.S. Government, nor the
//      names of its contributors may be used to endorse or promote
//      products derived from this software without specific prior
//      written permission.
//
//  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
//  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
//  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
//  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
//
// Tapir target that lowers to Kitsune's hip runtime
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TAPIR_HIP_ABI_H
#define LLVM_TRANSFORMS_TAPIR_HIP_ABI_H

#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"

#include <set>

namespace llvm {

class TapirTargetOptions;

/// The tapir target to lower tapir loops to kitsune's hip runtime. The tapir
/// loops will be converted to GPU kernels.
class HipABI : public TapirTarget {
public:
  HipABI(Module &HostM, const TapirTargetOptions &TTO);
  ~HipABI();

  /// Lower a call to the tapir.loop.grainsize intrinsic into a grain size
  /// (coarsening) value.  For GPU codes we currently limit this to a value of
  /// 1.
  Value *lowerGrainsizeCall(CallInst *GrainsizeCall) override final;

  /// Lower the given Tapir sync instruction (SI).
  void lowerSync(SyncInst &SI) override final;

  void preProcessModule() override final;

  /// Process Function F before any function outlining is performed.  This
  /// routine should not modify the CFG structure.
  bool preProcessFunction(Function &F, TaskInfo &TI,
                          bool ProcessingTapirLoops) override;

  // Add attributes to the Function Helper produced from outlining a task.
  void addHelperAttributes(Function &F) override;

  // Pre-process the Function F that has just been outlined from a task.  This
  // routine is executed on each outlined function by traversing in post-order
  // the tasks in the original function.
  void preProcessOutlinedTask(Function &F, Instruction *DetachPt,
                              Instruction *TaskFrameCreate, bool isSpawner,
                              BasicBlock *BB) override {}

  // Post-process the Function F that has just been outlined from a task.  This
  // routine is executed on each outlined function by traversing in post-order
  // the tasks in the original function.
  void postProcessOutlinedTask(Function &F, Instruction *DetachPtr,
                               Instruction *TaskFrameCreate, bool IsSpawner,
                               BasicBlock *TFEntry) override {}

  // Pre-process the root Function F as a function that can spawn subtasks.
  void preProcessRootSpawner(Function &F, BasicBlock *TFEntry) override {}

  // Post-process the root Function F as a function that can spawn subtasks.
  void postProcessRootSpawner(Function &F, BasicBlock *TFEntry) override {}

  // Process the invocation of a task for an outlined function.  This routine is
  // invoked after processSpawner once for each child subtask.
  void processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) override {}

  // Process Function F at the end of the lowering process.
  void postProcessFunction(Function &F, bool OutliningTapirLoops) override;

  // Process the host-side module at the end of lowering all functions //
  // within the module.
  void postProcessModule() override final;

  // Process a generated helper Function F produced via outlining, at the end of
  // the lowering process.
  void postProcessHelper(Function &F) override {}

  // Return the HIP outline processor associated with this target.
  LoopOutlineProcessor *
  getLoopOutlineProcessor(const TapirLoopInfo *TL) override final;

private:
  /// Currently, we create a single module into which all tapir loops to be
  /// run on an AMD GPU are outlined. A loop outline processor is created for
  /// each tapir loop which will add the outlined code into this module. This
  /// will eventually be compiled executable GPU code.
  Module KernelModule;

  /// When outlining tapir loops into the \ref KernelModule, we need to generate
  /// a name for the outlined function. This name must be unique. In the absence
  /// of debug information, the computed outlined function name consists of a
  /// fixed base with an integer suffix that is incremented for each tapir loop
  /// that is encountered.
  unsigned NextKernelID = 0;
};

/// The loop outline process for transforming a Tapir parallel loop into a
/// hip kernel function.
class HipLoop : public LoopOutlineProcessor {
public:
  /// @brief Build the HipLoop outline processor.
  /// @param M: Module containing the input code.
  /// @param KM: The module that will contain the generated kernel.
  /// @param KernelName: The name of the kernel function that is generated.
  /// @param TTO: The tapir target options.
  HipLoop(Module &M, Module &KM, StringRef KernelName,
          const TapirTargetOptions &TTO);
  ~HipLoop();

  /// Process the TapirLoop before it is outlined -- just prior to the
  /// outlining occurs.  This allows the VMap and related details to be
  /// customized prior to outlining related operations (e.g. cloning of
  /// LLVM constructs).
  void preProcessTapirLoop(TapirLoopInfo &TL, ValueToValueMapTy &VMap) override;

  /// Processes an outlined Function Helper for a Tapir loop, just after the
  /// function has been outlined.
  void postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                          ValueToValueMapTy &VMap) override;

  /// Processes a call to an outlined Function Helper for a Tapir loop.
  void processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                               DominatorTree &DT) override;

private:
  Value *emitWorkItemId(IRBuilder<> &Builder, int ItemIndex);
  Value *emitWorkGroupId(IRBuilder<> &Builder, int ItemIndex);
  Value *emitWorkGroupSize(IRBuilder<> &Builder, int ItemIndex);

  /// The name of the kernel into which the loop is outlined.
  std::string KernelName;

  /// For GPU targets, we outline the loop into a separate module. This is that
  /// module.
  Module &KernelModule;

  // AMDGCN intrinsics.
  FunctionCallee HipWorkItemIdFn;
  FunctionCallee HipWorkItemIdXFn, HipWorkItemIdYFn, HipWorkItemIdZFn;
  FunctionCallee HipWorkGroupIdFn;
  FunctionCallee HipWorkGroupIdXFn, HipWorkGroupIdYFn, HipWorkGroupIdZFn;
  FunctionCallee HipBlockDimFn;

  /// The GlobalValue's used in the loop that is being outlined. This includes
  /// functions, global variables, aliases and ifunc's.
  std::set<GlobalValue *> UsedGlobalValues;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_TAPIR_HIP_ABI_H

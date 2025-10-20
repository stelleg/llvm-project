//===- CudaABI.cpp - Lower Tapir Kitsune's hip runtime ------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Copyright (c) 2021, 2023, 2025 Los Alamos National Security, LLC.
//  All rights reserved.
//
// Copyright 2021, 2023, 2025. Los Alamos National Security, LLC. This
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
// Tapir target that lowers to Kitsune's cuda runtime
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/HipABI.h"
#include "kitsune/Core/CommandLineOptions.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/EmbUtils.h"
#include "kitsune/Core/KernelProperties.h"
#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Core/TapirTargetOptions.h"
#include "kitsune/Core/TargetUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/TapirLoopHints.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Tapir/KitsuneUtils.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "hipabi"

// For some background material see the AMDGPU target documentation
// at: https://llvm.org/docs/AMDGPUUsage.html
//
// This transformation is carrying out the prep to convert Tapir to a kernel
// module suitable for codegen using the AMDGPU target.

static cl::opt<unsigned> DefaultGrainSize(
    "hipabi-default-grainsize", cl::init(1), cl::Hidden,
    cl::desc("The default grain size used by the transform "
             "when analysis fails to determine one. (default=1)"),
    cl::cat(cl::catKitClDevOpts));

// FIXME: We really should not be exposing command line options from other
// source files. This is an experimental option that has been hacked in for the
// moment. If this is useful, we should consider adding it to the tapir target
// options instead. Otherwise, it should be removed altogether.
cl::opt<bool> clUseYLaunch("hipabi-y-launch", cl::init(false), cl::Hidden,
                           cl::desc("Launch kernel using y-axis threading"),
                           cl::cat(cl::catKitClDevOpts));

static constexpr StringRef HIPABI_PREFIX = "__kithip_";
static constexpr StringRef HIPABI_KERNEL_NAME_PREFIX = "__kithip_loop_";

/// @brief Return the work item ID for the calling thread. (thread index)
/// @param Builder - IR builder for code gen assistance.
/// @param ItemIndex - which work item dimension (x=0,y=1,z=2)
/// @param Low - Low-end of value range if known.
/// @param High -- High-end of value range if known.
Value *HipLoop::emitWorkItemId(IRBuilder<> &Builder, int ItemIndex) {
  switch (ItemIndex) {
  case 0:
    return Builder.CreateCall(HipWorkItemIdXFn, {}, ".kern.witem.x");
  case 1:
    return Builder.CreateCall(HipWorkItemIdYFn, {}, ".kern.witem.y");
  case 2:
    return Builder.CreateCall(HipWorkItemIdZFn, {}, ".kern.witem.z");
  default:
    llvm_unreachable("unexpected item index!");
  }
}

/// @brief Return the work group ID for the calling thread. (block index)
/// @param Builder - IR builder for code gen assistance.
/// @param ItemIndex - which work item dimension (x=0,y=1,z=2)
Value *HipLoop::emitWorkGroupId(IRBuilder<> &Builder, int ItemIndex) {
  switch (ItemIndex) {
  case 0:
    return Builder.CreateCall(HipWorkGroupIdXFn, {}, ".kern.wgroup.x");
  case 1:
    return Builder.CreateCall(HipWorkGroupIdYFn, {}, ".kern.wgroup.y");
  case 2:
    return Builder.CreateCall(HipWorkGroupIdZFn, {}, ".kern.wgroup.z");
  default:
    llvm_unreachable("unexpected item index!");
  }
}

/// @brief Return the work group size for the calling thread. (block size)
/// @param Builder - IR builder for code gen assistance.
/// @param ItemIndex - which work item dimension (x=0,y=1,z=2)
Value *HipLoop::emitWorkGroupSize(IRBuilder<> &Builder, int ItemIndex) {
  auto GetName = [](int ItemIndex) -> StringRef {
    switch (ItemIndex) {
    case 0:
      return ".kern.blkdim.x";
    case 1:
      return ".kern.blkdim.y";
    case 2:
      return ".kern.blkdim.z";
    default:
      llvm_unreachable("emitWorkGroupSize: unexpected item index!");
    };
  };

  LLVMContext &Ctx = Builder.getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Constant *Index = ConstantInt::get(Int32Ty, ItemIndex);
  StringRef Name = GetName(ItemIndex);
  return Builder.CreateCall(HipBlockDimFn, {Index}, Name);
}

HipLoop::HipLoop(Module &M, Module &KM, StringRef Name,
                 const TapirTargetOptions &TTO)
    : LoopOutlineProcessor(M, KM, TTO,
                           CloneFunctionChangeType::DifferentModule),
      KernelName(Name), KernelModule(KM) {
  LLVM_DEBUG(dbgs() << "hipabi: hip loop outliner creation:\n"
                    << "\ttransforming loop to kernel: " << KernelName
                    << "(...)\n"
                    << "\tdevice-side module name    : "
                    << KernelModule.getName() << "\n\n");

  LLVMContext &Ctx = KernelModule.getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);

  // We use ROCm/HSA/HIP entry points for various runtime calls.  These calls
  // are often at a lower level vs. user-facing entry points.  This follows
  // lower-level code generation details for HIP (that also include details
  // tucked into the HIP-centric header files as well a Clang lowering).

  // Get the local workitem ID for the calling thread.
  HipWorkItemIdFn = KernelModule.getOrInsertFunction(
      "__ockl_get_local_id",
      Int64Ty,  // return local thread id.
      Int32Ty); // axis/index select (x=0, y=1, z=2).

  // Get the work group ID for the calling thread.
  HipWorkGroupIdFn = KernelModule.getOrInsertFunction(
      "__ockl_get_group_id",
      Int64Ty,  // return local thread id.
      Int32Ty); // axis/index select (x=0, y=1, z=2).

  // Get the block size for the calling thread.
  HipBlockDimFn = KernelModule.getOrInsertFunction(
      "__ockl_get_local_size",
      Int64Ty,  // return local thread id.
      Int32Ty); // axis/index select (x=0, y=1, z=2).

  /* threadIdx.x */
  HipWorkItemIdXFn = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::amdgcn_workitem_id_x);
  /* threadIdx.y */
  HipWorkItemIdYFn = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::amdgcn_workitem_id_y);
  /* threadIdx. z */
  HipWorkItemIdZFn = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::amdgcn_workitem_id_z);

  /* blockIdx.x */
  HipWorkGroupIdXFn = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::amdgcn_workgroup_id_x);
  /* blockIdx.y */
  HipWorkGroupIdYFn = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::amdgcn_workgroup_id_y);
  /* blockIdx.z */
  HipWorkGroupIdZFn = Intrinsic::getOrInsertDeclaration(
      &KernelModule, Intrinsic::amdgcn_workgroup_id_z);
}

HipLoop::~HipLoop() { /* no-op */ }

void HipLoop::preProcessTapirLoop(TapirLoopInfo &TL, ValueToValueMapTy &VMap) {
  bool VerboseMode = getOptions().getTapirVerbose();
  if (VerboseMode) {
    errs() << "kitsune[hipabi]: pre-processing tapir loop.\n";
    errs() << "  - collecting global values from loop...\n";
  }

  // Collect the top-level entities (Function, GlobalVariable, GlobalAlias
  // and GlobalIFunc) that are used in the outlined loop. Since the outlined
  // loop will live in the KernelModule, any GlobalValue's used in it must be
  // be cloned into the KernelModule and then registered with the cuda runtime.
  // The registration will be done in the global ctor which will be generated by
  // a later pass.
  collectGlobalValues(*TL.getLoop(), UsedGlobalValues);

  // HIP appears to require protected visibility. Without this, attempting to
  // link the fat binary results in a relocation error.
  cloneUsedGlobalVariablesInto(
      KernelModule, UsedGlobalValues, VMap,
      /* address space for constant globals */ AMDGPUAS::CONSTANT_ADDRESS,
      /* address space for non-const globals */ AMDGPUAS::GLOBAL_ADDRESS,
      /* visibility for constant globals */ GlobalValue::DefaultVisibility,
      /* visibility for non-const globals */ GlobalValue::ProtectedVisibility);

  // The global variables have to be cloned before cloning the functions because
  // they may be used in the bodies of functions to be cloned.
  cloneReachableFuncsInto(KernelModule, UsedGlobalValues, VMap);
  cloneReachableIFuncsInto(KernelModule, UsedGlobalValues, VMap);

  // The aliasee in global aliases is a global value, so they must be cloned
  // after the global variables and functions are in the vmap.
  cloneUsedGlobalAliasesInto(KernelModule, UsedGlobalValues, VMap);
}

void HipLoop::postProcessOutline(TapirLoopInfo &TLI, TaskOutlineInfo &Out,
                                 ValueToValueMapTy &VMap) {
  Task *T = TLI.getTask();
  Loop *TL = TLI.getLoop();

  BasicBlock *Entry = cast<BasicBlock>(VMap[TL->getLoopPreheader()]);
  BasicBlock *Header = cast<BasicBlock>(VMap[TL->getHeader()]);
  BasicBlock *Exit = cast<BasicBlock>(VMap[TLI.getExitBlock()]);
  PHINode *PrimaryIV = cast<PHINode>(VMap[TLI.getPrimaryInduction().first]);
  Value *PrimaryIVInput = PrimaryIV->getIncomingValueForBlock(Entry);
  Type *PrimaryIVType = PrimaryIV->getType();

  // We no longer need the cloned sync region.
  auto *ClonedSyncReg =
      cast<Instruction>(VMap[T->getDetach()->getSyncRegion()]);
  ClonedSyncReg->eraseFromParent();

  // Get the kernel function for this loop and clean up any stray (target
  // related) attributes. Because of the way we compile the code, those
  // attributes will only be relevant for the host.
  Function *KernelF = Out.Outline;
  KernelF->setName(KernelName);

  // Remove any attributes that are only relevant for the host.
  KernelF->removeFnAttr("target-cpu");
  KernelF->removeFnAttr("target-features");
  KernelF->removeFnAttr("tune-cpu");

  // Remove other attributes that we cannot deal with in any reasonable way in
  // the device
  KernelF->removeFnAttr(Attribute::UWTable);

  // Add an attribute identifying this as a function outlined from a tapir loop.
  KernelF->addFnAttr(Attribute::KitKernel);

  // Add new target-specific attributes
  KernelF->addFnAttr("target-cpu", getOptions().getHipArch());
  KernelF->addFnAttr("target-features", getOptions().getHipTargetFeatures());

  // Add other attributes that are either required or desirable.
  KernelF->addFnAttr("no-trapping-math", "true");
  KernelF->addFnAttr(Attribute::MustProgress);
  KernelF->addFnAttr(Attribute::NoUnwind);

  // This only works when the code object version >= 5, but we have ensured that
  // this is the case in the frontend.
  KernelF->addFnAttr("uniform-work-group-size", "true");

  // Specify the minimum and maximum flat work group sizes that will be used
  // when the kernel is dispatched.
  std::string AttrVal;
  AttrVal = std::string("128,1024");
  KernelF->addFnAttr("amdgpu-flat-work-group-size", AttrVal);

  // AMD requires that the kernel function have protected visiblity otherwise
  // AMD's runtime is unable to find the kernel function at runtime. This, in
  // turn requires the function to have external linkage. In case the function
  // gets here with a different linkage type, just override it.
  KernelF->setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);
  KernelF->setVisibility(GlobalValue::VisibilityTypes::ProtectedVisibility);
  KernelF->setCallingConv(CallingConv::AMDGPU_KERNEL);

#if 0 // DISABLED FOR TESTING
  unsigned MaxThreadsPerBlock = getOptions().getMaxThreadsPerBlock();
  unsigned DefaultThreadsPerBlock = getOptions().getFixedThreadsPerBlock();

  // Check for programmer-provided launch attribute...
  if (TPB > 0 && TPB <= MaxThreadsPerBlock) {
    AttrVal = std::string("1,") + utostr(TPB);
    KernelF->addFnAttr("amdgpu-flat-work-group-size", AttrVal);

    if (clUseYLaunch)
      AttrVal = std::string("1,") + utostr(TPB) + std::string(",1");
    else
      AttrVal = utostr(TPB) + std::string(",1,1");
  } else if (DefaultThreadsPerBlock > 0 &&
             DefaultThreadsPerBlock <= MaxThreadsPerBlock) {
    // Check for command line spec.
    AttrVal = std::string("1,") + utostr(DefaultThreadsPerBlock);
    KernelF->addFnAttr("amdgpu-flat-work-group-size", AttrVal);

    if (clUseYLaunch)
      AttrVal = std::string("1,") + utostr(DefaultThreadsPerBlock)
                    + std::string(",1");
    else
      AttrVal = utostr(DefaultThreadsPerBlock) + std::string(",1,1");
  } else {
    // Use defaults...
    AttrVal = std::string("1,") + utostr(MaxThreadsPerBlock);
    KernelF->addFnAttr("amdgpu-flat-work-group-size", AttrVal);

    if (clUseYLaunch)
      AttrVal = std::string("1,") + utostr(MaxThreadsPerBlock) +
                std::string(",1");
    else
      AttrVal = utostr(MaxThreadsPerBlock) + std::string(",1,1");
  }
  // Attribute falls through from above conditionals...
  KernelF->addFnAttr("amdgpu-max-num-workgroups", AttrVal);
#endif

  // Verify that the Thread ID corresponds to a valid iteration. Because Tapir
  // loops use canonical induction variables, valid iterations range from 0 to
  // the loop limit with stride 1. The End argument encodes the limit.
  // Get end and grain size arguments

  // End argument is always the second argument in the kernel function.
  Argument *End = KernelF->getArg(1);

  // Get the grainsize value, which is either constant or the third LC arg.
  // TODO: We only support a grain size of 1 right now. Not clear if this
  // could be a future optimization but strip mining on our current tests only
  // results in degraded performance.
  // if (unsigned ConstGrainsize = TLI.getGrainsize())
  //  Grainsize = ConstantInt::get(PrimaryIV->getType(), ConstGrainsize);
  // else
  Value *Grainsize =
      ConstantInt::get(PrimaryIV->getType(), DefaultGrainSize.getValue());

  IRBuilder<> Builder(Entry->getTerminator());

  // Get the thread ID for this invocation of Helper.
  //
  // This is the classic thread ID calculation:
  //      i = blockDim.x * blockIdx.x + threadIdx.x;
  // For now we only generate 1-D thread IDs.
  Value *ThreadIdx;
  Value *BlockDim;

  if (not clUseYLaunch) {
    Value *WorkItemIdX = emitWorkItemId(Builder, 0);
    ThreadIdx = Builder.CreateIntCast(WorkItemIdX, PrimaryIVType,
                                      /*isSigned=*/false, ".kern.tidx.x");
    Value *WorkGroupSizeX = emitWorkGroupSize(Builder, 0);
    BlockDim = Builder.CreateIntCast(WorkGroupSizeX, PrimaryIVType,
                                     /*isSigned=*/false, ".kern.blkdim.x");
  } else {
    Value *WorkItemIdY = emitWorkItemId(Builder, 1);
    ThreadIdx = Builder.CreateIntCast(WorkItemIdY, PrimaryIVType,
                                      /*isSigned=*/false, ".kern.tidx.y");
    Value *WorkGroupSizeY = emitWorkGroupSize(Builder, 1);
    BlockDim = Builder.CreateIntCast(WorkGroupSizeY, PrimaryIVType,
                                     /*isSigned=*/false, ".kern.blkdim.y");
  }

  Value *WorkGroupIdX = emitWorkGroupId(Builder, 0);
  Value *BlockIdx = Builder.CreateIntCast(WorkGroupIdX, PrimaryIVType,
                                          /*isSigned=*/false, ".kern.blkid.x");

  Value *BlockOff = Builder.CreateMul(BlockIdx, BlockDim, ".kern.blkoff.x");
  Value *ThreadID = Builder.CreateAdd(ThreadIdx, BlockOff, ".kern.tid");

  // NOTE/TODO: Assuming that the grainsize is fixed at 1 for the current
  // codegen.
  // ThreadID = Builder.CreateMul(ThreadID, Grainsize);
  Value *ThreadEnd = Builder.CreateAdd(ThreadID, Grainsize, ".kern.last_idx");
  Value *Cond = Builder.CreateICmpUGE(ThreadID, End, ".kern.at_end");
  ReplaceInstWithInst(Entry->getTerminator(),
                      BranchInst::Create(Exit, Header, Cond));

  // Replace the loop's induction variable with the GPU thread id.
  PrimaryIVInput->replaceAllUsesWith(ThreadID);

  // Update cloned loop condition to use the thread-end value.
  unsigned TripCountIdx = 0;
  ICmpInst *ClonedCond = cast<ICmpInst>(VMap[TLI.getCondition()]);
  if (ClonedCond->getOperand(0) != End)
    ++TripCountIdx;
  assert(ClonedCond->getOperand(TripCountIdx) == End &&
         "End argument not used in condition!");
  ClonedCond->setOperand(TripCountIdx, ThreadEnd);
}

void HipLoop::processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                                      DominatorTree &DT) {
  LLVM_DEBUG(dbgs() << "hiploop: processing outlined loop call...\n"
                    << "\tkernel name: " << KernelName << "\n");

  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  PointerType *PtrTy = PointerType::getUnqual(Ctx);

  ConstantInt *CTT = createConstInt(TTID::Hip, Ctx);
  GlobalVariable *KProps =
      createKernelPropertiesGlobal(KernelName, TTID::Hip, M);
  Value *KName = createConstString(KernelName, M);
  GlobalVariable *EmbFB = getEmbFBGlobal(TTID::Hip, M);

  // At this point we need a threads-per-block value for the launch call. The
  // runtime will determine this value if ThreadsPerBlock is zero but it can
  // also be overridden via kitsune's forall launch attribute. The catch here is
  // the launch attribute's value for this is flexible and be a computed
  // expression vs. a compile-time constant. For this first step of creating the
  // kernel launch, we take the path of a runtime configuration vs. an
  // attributed launch.
  TapirLoopHints Hints(TL.getLoop());
  unsigned TPBHint = Hints.getThreadsPerBlock();
  unsigned FixedThreadsPerBlock = getOptions().getFixedThreadsPerBlock();
  Value *TPB;
  if (TPBHint)
    TPB = ConstantInt::get(Int32Ty, TPBHint);
  else if (FixedThreadsPerBlock)
    TPB = ConstantInt::get(Int32Ty, FixedThreadsPerBlock);
  else
    TPB = ConstantInt::get(Int32Ty, 0);

  CallBase *CallOutlined = cast<CallBase>(TOI.ReplCall);
  BasicBlock *RCBB = CallOutlined->getParent();
  BasicBlock *NewBB = RCBB->splitBasicBlock(CallOutlined);
  IRBuilder<> Builder(&NewBB->front());

  // Deal with type mismatches for the trip count.
  Value *TripCount = CallOutlined->getArgOperand(1);
  if (TripCount->getType() != Int64Ty)
    TripCount = Builder.CreateSExtOrBitCast(TripCount, Int64Ty, "cast.tc");

  // We need to explicitly sync non-const globals that are used in the kernel
  // before the kernel is launched.
  copyNonConstGlobalsHToD(UsedGlobalValues, TTID::Hip, M, Builder);

  Value *HipStream =
      Builder.CreateIntrinsic(PtrTy, Intrinsic::kit_thread_stream, {CTT});
  std::vector<Value *> Args = {CTT, EmbFB,  KName,    TripCount,
                               TPB, KProps, HipStream};
  for (Value *Inp : CallOutlined->args())
    Args.push_back(Inp);

  // TODO: We should probably have the launch and sync kitsune intrinsics take
  // a sync region as an argument This may make it easier to do post-outlining
  // analyses to eliminate/delay device synchronization calls instead of
  // always synchronizing immediately after the kernel launch.
  LLVM_DEBUG(dbgs() << "\t*- code gen kernel launch....\n");
  (void)Builder.CreateCall(
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::kit_async_launch_kernel),
      Args);
  (void)Builder.CreateIntrinsic(VoidTy, Intrinsic::kit_sync_stream,
                                {CTT, HipStream});

  // After the kernel is done, copy the non-const globals back to the host. This
  // is done here to keep this part of the code generation simple. A subsequent
  // pass will attempt to move this call to the point where the global is
  // actually used on the host (or perhaps even delete it if the host never uses
  // the global again).
  copyNonConstGlobalsDToH(UsedGlobalValues, TTID::Hip, M, Builder);

  CallOutlined->eraseFromParent();
  LLVM_DEBUG(dbgs() << "*** finished processing outlined call.\n");
}

// As is the pattern with the GPU targets, the HipABI is setup to process all
// Tapir constructs within a given input Module (M). It then creates a
// corresponding module that contains the transformed device-side code. This is
// the KernelModule that is created below in the target constructor.
HipABI::HipABI(Module &M, const TapirTargetOptions &TTO)
    : TapirTarget(M, TTO), KernelModule("", M.getContext()), NextKernelID(0) {
  LLVM_DEBUG(dbgs() << "hipABI: HipABI::HipABI()\n");

  TargetMachine *TM = createTargetMachine(TTID::Hip, TTO);
  KernelModule.setTargetTriple(TM->getTargetTriple());
  KernelModule.setDataLayout(TM->createDataLayout());

  KernelModule.setModuleIdentifier(getNameForDeviceModule(M, HIPABI_PREFIX));
  addDeviceModuleMetadata(TTID::Hip, KernelModule);
  cloneModuleFlagsMetadataInto(M, KernelModule);
  cloneIdentMetadataInto(M, KernelModule);
}

HipABI::~HipABI() { /* no-op */ }

Value *HipABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
  // TODO: The grain size on the GPU is a completely different beast than the
  // CPU cases Tapir was originally designed for. At present keeping the grain
  // size at 1 has almost always shown to yield the best results in terms of
  // performance but we should take a closer look...  We have some tweaks for
  // experimenting with this via the command line but it remains unexplored.
  Value *Grainsize;
  Grainsize = ConstantInt::get(GrainsizeCall->getType(), DefaultGrainSize);
  // Replace uses of grain size intrinsic call with a computed grain size value.
  GrainsizeCall->replaceAllUsesWith(Grainsize);
  GrainsizeCall->eraseFromParent();
  return Grainsize;
}

void HipABI::lowerSync(SyncInst &SI) {}

void HipABI::addHelperAttributes(Function &F) {}

void HipABI::preProcessModule() {
  // Create the global variable that will eventually contain the fat binary of
  // GPU code. This is currently uninitialized, but will be passed to several
  // of the kitsune runtime intrinsic calls when launching kernels, copying
  // global variables from host to device etc.
  (void)createEmbFBGlobal(TTID::Hip, M);
}

bool HipABI::preProcessFunction(Function &F, TaskInfo &TI,
                                bool OutliningTapirLoops) {
  return false;
}

void HipABI::postProcessFunction(Function &F, bool OutliningTapirLoops) {}

void HipABI::postProcessModule() {
  // At this point, we are done with the minimum task of outlining the tapir
  // loop into a kernel module. There are still a number of transformations that
  // must be carried out on this module before it can be compiled to GPU code,
  // but those will be done by subsequent passes. The module here is in a state
  // where we can perform combined host/device analyses and optimizations.
  (void)createEmbBCGlobal(KernelModule, TTID::Hip, M);
}

LoopOutlineProcessor *HipABI::getLoopOutlineProcessor(const TapirLoopInfo *TL) {
  LLVM_DEBUG(dbgs() << "hipabi: create loop outlining processor.\n");
  LLVM_DEBUG(saveModuleToFile(&M, M.getName().str() + ".input"));

  std::string KernelName =
      getNameForTapirLoop(*TL, HIPABI_KERNEL_NAME_PREFIX, NextKernelID++);
  return new HipLoop(M, KernelModule, KernelName, this->getOptions());
}

//===- HIPABI.cpp - Lower Tapir loop to a HIP kernel --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the HIP ABI to convert Tapir instructions to calls into
// GPU runtime system for NVIDIA GPU code.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/HIPABI.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Utils/TapirUtils.h"
#include "llvm/Transforms/Vectorize.h"

using namespace llvm;

#define DEBUG_TYPE "hipabi"

const unsigned ThreadsPerBlock = 1024;

static cl::opt<bool>
Verbose("tapir-hip-verbose", cl::init(false),
        cl::desc("Verbose output for Tapir-HIP backend"));

void AMDGCNLoop::EmitAMDGCN() {
  legacy::PassManager PM;
  legacy::FunctionPassManager FPM(&AMDGCNM);

	LLVM_DEBUG(dbgs() << "AMDGCN Module before optimizations:" << AMDGCNM); 

  PassManagerBuilder Builder;
  Builder.OptLevel = CodeGenOpt::Default;
  Builder.SizeLevel = 0;
  AMDGCNTargetMachine->adjustPassManager(Builder);
	Builder.populateModulePassManager(PM); 
  Builder.populateFunctionPassManager(FPM);

  SmallVector<char, 65536> buf; 
  raw_svector_ostream ostr(buf); 

  FPM.doInitialization();
  for (Function &F : AMDGCNM)
    FPM.run(F);
  FPM.doFinalization();
  PM.add(createVerifierPass());
  //PM.run(AMDGCNM);
  bool Fail = AMDGCNTargetMachine->addPassesToEmitFile(
      PM, ostr, nullptr,
      CodeGenFileType::CGFT_ObjectFile, false);
  assert(!Fail && "Failed to emit AMDGCN");
  // Add function optimization passes.

	LLVM_DEBUG(dbgs() << "AMDGCN Module after optimizations, before writing to buffer" << AMDGCNM); 

  std::string hsaco = ostr.str().str(); 
  Constant* cda = ConstantDataArray::getString(M.getContext(), hsaco); 
  AMDGCNGlobal = new GlobalVariable(M, cda->getType(), true, GlobalValue::PrivateLinkage, cda, GlobalName);  
}

Value *HIPABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
  Value *Grainsize = ConstantInt::get(GrainsizeCall->getType(), 8);

  // Replace uses of grainsize intrinsic call with this grainsize value.
  GrainsizeCall->replaceAllUsesWith(Grainsize);
  return Grainsize;
}

void HIPABI::lowerSync(SyncInst &SI) {
  // currently a no-op...
}

void HIPABI::preProcessFunction(Function &F, TaskInfo &TI,
                                 bool OutliningTapirLoops) {
}

void HIPABI::postProcessFunction(Function &F, bool OutliningTapirLoops) {
  if (!OutliningTapirLoops || !LOP)
    return;

  LOP->EmitAMDGCN();
}

void HIPABI::postProcessHelper(Function &F) {
}

void HIPABI::preProcessOutlinedTask(Function &F, Instruction *DetachPt,
                                     Instruction *TaskFrameCreate,
                                     bool IsSpawner) {
}

void HIPABI::postProcessOutlinedTask(Function &F, Instruction *DetachPt,
                                      Instruction *TaskFrameCreate,
                                      bool IsSpawner) {
}

void HIPABI::preProcessRootSpawner(Function &F) {
}

void HIPABI::postProcessRootSpawner(Function &F) {
}

void HIPABI::processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) {
}

LoopOutlineProcessor *HIPABI::getLoopOutlineProcessor(
    const TapirLoopInfo *TL) {
  if (!LOP)
    LOP = new HIPLoop(M);
  return LOP;
}

// Static counter for assigning IDs to kernels.
unsigned AMDGCNLoop::NextKernelID = 0;

AMDGCNLoop::AMDGCNLoop(Module &M)
    : LoopOutlineProcessor(M, AMDGCNM),
      AMDGCNM("amdgcn module", M.getContext()) {
  // Assign an ID to this kernel.
  MyKernelID = NextKernelID++;

  // Setup an NVAMDGCN triple.
  Triple AMDGCNTriple("amdgcn", "amd", "amdhsa");
  AMDGCNM.setTargetTriple(AMDGCNTriple.str());

  // Find the NVAMDGCN module pass which will create the AMDGCN code
  std::string error;
  const Target *AMDGCNTarget = TargetRegistry::lookupTarget(AMDGCNTriple.str(), error);
  LLVM_DEBUG({
      if (!AMDGCNTarget)
        dbgs() << "ERROR: Failed to lookup AMDGCN target: " << error << "\n";
    });
  assert(AMDGCNTarget && "Failed to find AMDGCN target");

  AMDGCNTargetMachine =
      AMDGCNTarget->createTargetMachine(AMDGCNTriple.getTriple(), "gfx900",
                                     "", TargetOptions(), Reloc::PIC_,
                                     CodeModel::Small, CodeGenOpt::Aggressive);
  AMDGCNM.setDataLayout(AMDGCNTargetMachine->createDataLayout());

  // Insert runtime-function declarations in AMDGCN host modules.
  Type *AMDGCNInt32Ty = Type::getInt32Ty(AMDGCNM.getContext());
  GetThreadId = AMDGCNM.getOrInsertFunction("hc_get_workitem_id", AMDGCNInt32Ty, AMDGCNInt32Ty); 
  GetBlockId = AMDGCNM.getOrInsertFunction("hc_get_group_id", AMDGCNInt32Ty, AMDGCNInt32Ty); 
  GetBlockDim = AMDGCNM.getOrInsertFunction("hc_get_group_size", AMDGCNInt32Ty, AMDGCNInt32Ty); 
}

void AMDGCNLoop::setupLoopOutlineArgs(
    Function &F, ValueSet &HelperArgs, SmallVectorImpl<Value *> &HelperInputs,
    ValueSet &InputSet, const SmallVectorImpl<Value *> &LCArgs,
    const SmallVectorImpl<Value *> &LCInputs, const ValueSet &TLInputsFixed) {
  // Add the loop control inputs.

  // The first parameter defines the extent of the index space, i.e., the number
  // of threads to launch.
  {
    Argument *EndArg = cast<Argument>(LCArgs[1]);
    EndArg->setName("runSize");
    HelperArgs.insert(EndArg);

    Value *InputVal = LCInputs[1];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }
  // The second parameter defines the start of the index space.
  {
    Argument *StartArg = cast<Argument>(LCArgs[0]);
    StartArg->setName("runStart");
    HelperArgs.insert(StartArg);

    Value *InputVal = LCInputs[0];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }
  // The third parameter defines the grainsize, if it is not constant.
  if (!isa<ConstantInt>(LCInputs[2])) {
    Argument *GrainsizeArg = cast<Argument>(LCArgs[2]);
    GrainsizeArg->setName("runStride");
    HelperArgs.insert(GrainsizeArg);

    Value *InputVal = LCInputs[2];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }

  // Add the remaining inputs
  for (Value *V : TLInputsFixed) {
    assert(!HelperArgs.count(V));
    HelperArgs.insert(V);
    HelperInputs.push_back(V);
  }
}

unsigned AMDGCNLoop::getIVArgIndex(const Function &F, const ValueSet &Args) const {
  // The argument for the primary induction variable is the second input.
  return 1;
}

unsigned AMDGCNLoop::getLimitArgIndex(const Function &F, const ValueSet &Args)
  const {
  // The argument for the loop limit is the first input.
  return 0;
}

void AMDGCNLoop::postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                                 ValueToValueMapTy &VMap) {
  Task *T = TL.getTask();
  Loop *L = TL.getLoop();

  Function *Helper = Out.Outline;
  BasicBlock *Entry = cast<BasicBlock>(VMap[L->getLoopPreheader()]);
  BasicBlock *Header = cast<BasicBlock>(VMap[L->getHeader()]);
  BasicBlock *Exit = cast<BasicBlock>(VMap[TL.getExitBlock()]);
  PHINode *PrimaryIV = cast<PHINode>(VMap[TL.getPrimaryInduction().first]);
  Value *PrimaryIVInput = PrimaryIV->getIncomingValueForBlock(Entry);
  Instruction *ClonedSyncReg = cast<Instruction>(
      VMap[T->getDetach()->getSyncRegion()]);

  // We no longer need the cloned sync region.
  ClonedSyncReg->eraseFromParent();

  // Set the helper function to have external linkage.
  Helper->setLinkage(Function::ExternalLinkage);

  // Verify that the Thread ID corresponds to a valid iteration.  Because Tapir
  // loops use canonical induction variables, valid iterations range from 0 to
  // the loop limit with stride 1.  The End argument encodes the loop limit.
  // Get end and grainsize arguments
  Value *Grainsize;
  auto OutlineArgsIter = Helper->arg_begin();
    // End argument is the first LC arg.
  Argument *End = &*OutlineArgsIter++;
    // Start argument is the second LC arg.
  Argument *Start = &*OutlineArgsIter++;

  // Get the grainsize value, which is either constant or the third LC arg.
  if (unsigned ConstGrainsize = TL.getGrainsize())
    Grainsize = ConstantInt::get(PrimaryIV->getType(), ConstGrainsize);
  else
    // Grainsize argument is the third LC arg.
    Grainsize = &*OutlineArgsIter;

  // Get the thread ID for this invocation of Helper.
  LLVMContext &Ctx = AMDGCNM.getContext();
  IRBuilder<> B(Entry->getTerminator());
  Value *ThreadIdx = B.CreateCall(GetThreadId, ConstantInt::get(Type::getInt32Ty(Ctx), 0));
  Value *BlockIdx = B.CreateCall(GetBlockId, ConstantInt::get(Type::getInt32Ty(Ctx), 0));
  Value *BlockDim = B.CreateCall(GetBlockDim, ConstantInt::get(Type::getInt32Ty(Ctx), 0));
  Value *ThreadID = B.CreateIntCast(
      B.CreateAdd(ThreadIdx, B.CreateMul(BlockIdx, BlockDim), "threadId"),
      PrimaryIV->getType(), false);

  ThreadID = B.CreateMul(ThreadID, Grainsize);
  Value *ThreadEnd = B.CreateAdd(ThreadID, Grainsize);
  Value *Cond = B.CreateICmpUGE(ThreadID, End);

  ReplaceInstWithInst(Entry->getTerminator(), BranchInst::Create(Exit, Header,
                                                                 Cond));
  // Use the thread ID as the start iteration number for the primary IV.
  PrimaryIVInput->replaceAllUsesWith(ThreadID);

  // Update cloned loop condition to use the thread-end value.
  unsigned TripCountIdx = 0;
  ICmpInst *ClonedCond = cast<ICmpInst>(VMap[TL.getCondition()]);
  if (ClonedCond->getOperand(0) != End)
    ++TripCountIdx;
  assert(ClonedCond->getOperand(TripCountIdx) == End &&
         "End argument not used in condition");
  ClonedCond->setOperand(TripCountIdx, ThreadEnd);
}

HIPLoop::HIPLoop(Module &M) : AMDGCNLoop(M) {
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *SizeTy = DL.getIntPtrType(Ctx);
  Type *VoidTy = Type::getVoidTy(Ctx); 
  PointerType *CharPtrTy = Type::getInt8PtrTy(Ctx);
  
  // We use kitsune wrapper, as hip calls use C++ types, and fuck that
  // kitsuneLaunchKernel 
  //  (kernel : char*) (kernelname : char*) (args : char*) (argsize : size)
  //  (nthread : size) (nblock : size) : void
  KitsuneLaunchKernel = M.getOrInsertFunction(
    "kitsuneLaunchKernel", VoidTy, CharPtrTy, CharPtrTy, CharPtrTy, SizeTy, SizeTy, SizeTy); 
}

void HIPLoop::processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                                       DominatorTree &DT) {
  Function *Outlined = TOI.Outline;
  Instruction *ReplStart = TOI.ReplStart;
  Instruction *ReplCall = TOI.ReplCall;
  CallSite CS(ReplCall);
  BasicBlock *CallCont = TOI.ReplRet;
  BasicBlock *UnwindDest = TOI.ReplUnwind;
  Function *Parent = ReplCall->getFunction();
  //Value *TripCount = CS.getArgOperand(getLimitArgIndex(*Parent, TOI.InputSet));
  LLVM_DEBUG(dbgs() << "Post processing AMDGCN Module: " << AMDGCNM); 

  LLVMContext &Ctx = M.getContext();
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);

  // Split the basic block containing the detach replacement just before the
  // start of the detach-replacement instructions.
  BasicBlock *DetBlock = ReplStart->getParent();
  BasicBlock *CallBlock = SplitBlock(DetBlock, ReplStart);

  // Create a call to HIPPopCallConfiguration
  IRBuilder<> B(ReplCall);

  // Create an array of kernel arguments.
  AllocaInst *KernelArgs;
  // Calculate amount of space we will need for all arguments.  If we have no
  // args, allocate a single pointer so we still have a valid pointer to the
  // argument array that we can pass to runtime, even if it will be unused.
  if (CS.arg_empty())
    // CS.args() contains no arguments to pass to the kernel.
    KernelArgs = B.CreateAlloca(VoidPtrTy, nullptr, "kernel_args");
  else {
    KernelArgs = B.CreateAlloca(VoidPtrTy, B.getInt8(CS.arg_size()),
                                "kernel_args");
    // Store pointers to the arguments.
    unsigned Index = 0, ArgNum = 0;
    for (Value *Arg : CS.args()) {
      AllocaInst *ArgAlloc = B.CreateAlloca(Arg->getType());
      B.CreateStore(Arg, ArgAlloc);
      B.CreateStore(
          B.CreateBitCast(ArgAlloc, VoidPtrTy),
          B.CreateConstInBoundsGEP1_32(VoidPtrTy, KernelArgs, Index));
      ++Index; ++ArgNum;
    }
  }
	
 //hip
	
  // Update the value of ReplCall.
  ReplCall = TOI.ReplCall;

  // hipLaunchKernel needs this function to have the same type as the kernel
  ValueSet SHInputs;
  for (Value *Arg : CS.args())
    SHInputs.insert(Arg);

  ValueSet Outputs;  // Should be empty.
  // Only one block needs to be cloned into the spawn helper
  std::vector<BasicBlock *> BlocksToClone;
  BlocksToClone.push_back(CallBlock);
  SmallVector<ReturnInst *, 1> Returns;  // Ignore returns cloned.
  ValueToValueMapTy VMap;
  Twine NameSuffix = ".stub";
  Function *HIPLoopHelper =
      CreateHelper(SHInputs, Outputs, BlocksToClone, CallBlock, DetBlock,
                   CallCont, VMap, &M, Parent->getSubprogram() != nullptr,
                   Returns, NameSuffix.str(), nullptr, nullptr, nullptr,
                   UnwindDest);

  assert(Returns.empty() && "Returns cloned when creating SpawnHelper.");

  // If there is no unwind destination, then the SpawnHelper cannot throw.
  if (!UnwindDest)
    HIPLoopHelper->setDoesNotThrow();

  // Add attributes to new helper function.
  HIPLoopHelper->setUnnamedAddr(GlobalValue::UnnamedAddr::None);

  // hipLaunchKernel needs this function to have the same name as the kernel
  HIPLoopHelper->setName(Outlined->getName());

	
  // Add alignment assumptions to arguments of helper, based on alignment of
  // values in old function.
  AddAlignmentAssumptions(Parent, SHInputs, VMap, ReplCall, nullptr, nullptr);

  // Move allocas in the newly cloned block to the entry block of the helper.
  {
    // Collect the end instructions of the task.
    SmallVector<Instruction *, 4> Ends;
    Ends.push_back(cast<BasicBlock>(VMap[CallCont])->getTerminator());
    if (isa<InvokeInst>(ReplCall))
      Ends.push_back(cast<BasicBlock>(VMap[UnwindDest])->getTerminator());

    // Move allocas in cloned detached block to entry of helper function.
    BasicBlock *ClonedBlock = cast<BasicBlock>(VMap[CallBlock]);
    MoveStaticAllocasInBlock(&HIPLoopHelper->getEntryBlock(), ClonedBlock,
                             Ends);

    // We do not need to add new llvm.stacksave/llvm.stackrestore intrinsics,
    // because calling and returning from the helper will automatically manage
    // the stack appropriately.
  }

  // Insert a call to the spawn helper.
  SmallVector<Value *, 8> SHInputVec;
  for (Value *V : SHInputs)
    SHInputVec.push_back(V);
  SplitEdge(DetBlock, CallBlock);
  B.SetInsertPoint(CallBlock->getTerminator());
  /* 
  Value *ModVal = B.CreateAnd(TripCount, ThreadsPerBlock, "xtraiter");
  Value *BranchVal = B.CreateICmpULT(TripCount,
                                     ConstantInt::get(TripCount->getType(),
                                                      ThreadsPerBlock));
  Value *BlockVal = B.CreateSelect(BranchVal, TripCount,
                                   ConstantInt::get(TripCount->getType(),
                                                    ThreadsPerBlock));
  Value *Threads = B.CreateAdd(
      ConstantInt::get(TripCount->getType(), 1),
      B.CreateUDiv(TripCount, ConstantInt::get(TripCount->getType(),
                                               ThreadsPerBlock)));

  */
  // TODO: Use LoopStripMine to generate specialized versions of the kernel for
  // different iteration counts, rather than use a complex call configuration.
  if (isa<InvokeInst>(ReplCall)) {
    InvokeInst *HelperCall = InvokeInst::Create(HIPLoopHelper, CallCont,
                                                UnwindDest, SHInputVec);
    HelperCall->setDebugLoc(ReplCall->getDebugLoc());
    HelperCall->setCallingConv(HIPLoopHelper->getCallingConv());
    ReplaceInstWithInst(CallBlock->getTerminator(), HelperCall);
  } else {
    CallInst *HelperCall = B.CreateCall(HIPLoopHelper, SHInputVec);
    HelperCall->setDebugLoc(ReplCall->getDebugLoc());
    HelperCall->setCallingConv(HIPLoopHelper->getCallingConv());
    HelperCall->setDoesNotThrow();
    // Branch around CallBlock.  Its contents are now dead.
    ReplaceInstWithInst(CallBlock->getTerminator(),
                        BranchInst::Create(CallCont));
  }

  // Erase extraneous call instructions
  cast<Instruction>(VMap[ReplCall])->eraseFromParent();
  ReplCall->eraseFromParent();
}

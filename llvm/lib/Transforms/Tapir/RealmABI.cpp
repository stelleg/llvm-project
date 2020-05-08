//===- RealmABI.cpp - Lower Tapir into Realm runtime system calls -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the RealmABI interface, which is used to convert Tapir
// instructions -- detach, reattach, and sync -- to calls into the Realm
// runtime system.  
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/RealmABI.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "realmabi"

FunctionCallee RealmABI::get_realmGetNumProcs() {
  if(RealmGetNumProcs)
    return RealmGetNumProcs;

  LLVMContext &C = M.getContext(); 
  const DataLayout &DL = M.getDataLayout();
  AttributeList AL;
  // TODO: Set appropriate function attributes.
  FunctionType *FTy = FunctionType::get(DL.getIntPtrType(C), {}, false);
  RealmGetNumProcs = M.getOrInsertFunction("realmGetNumProcs", FTy, AL);
  return RealmGetNumProcs;
}

static StructType* getBarrierType(LLVMContext &C){
  auto eventTy = StructType::get(Type::getInt64Ty(C));
  return StructType::get(eventTy, Type::getInt64Ty(C));
}

FunctionCallee RealmABI::get_createRealmBarrier(){
  if(CreateBar) return CreateBar; 
  LLVMContext &C = M.getContext(); 

  AttributeList AL; 
  FunctionType *FTy = FunctionType::get(
    getBarrierType(C), {}, false);
  CreateBar = M.getOrInsertFunction("createRealmBarrier", FTy, AL);
}

FunctionCallee RealmABI::get_destroyRealmBarrier(){
  if(DestroyBar) return DestroyBar; 
  LLVMContext &C = M.getContext(); 

  AttributeList AL; 
  FunctionType *FTy = FunctionType::get(
    Type::getVoidTy(C), {getBarrierType(C)}, false);
  DestroyBar = M.getOrInsertFunction("destroyRealmBarrier", FTy, AL);
}

FunctionCallee RealmABI::get_realmSpawn() {
  if(RealmSpawn)
    return RealmSpawn;

  LLVMContext &C = M.getContext(); 
  const DataLayout &DL = M.getDataLayout();
  AttributeList AL;
  // TODO: Set appropriate function attributes.
  FunctionType *FTy = FunctionType::get(
      Type::getInt32Ty(C),     // returns int 
      { TaskFuncPtrTy,         // TaskFuncPtr fxn
        Type::getInt8PtrTy(C), // const void *args
        DL.getIntPtrType(C),   // size_t arglen
        getBarrierType(C)
      }, false);
  RealmSpawn = M.getOrInsertFunction("realmSpawn", FTy, AL);
  return RealmSpawn;
}

FunctionCallee RealmABI::get_realmSync() {
  if(RealmSync)
    return RealmSync;

  LLVMContext &C = M.getContext(); 
  AttributeList AL;
  // TODO: Set appropriate function attributes.
  FunctionType *FTy = FunctionType::get(Type::getInt32Ty(C), {
    getBarrierType(C)
    }, false);
  RealmSync = M.getOrInsertFunction("realmSync", FTy, AL);
  return RealmSync;
}

FunctionCallee RealmABI::get_realmInitRuntime() {
  if(RealmInitRuntime)
    return RealmInitRuntime;

  LLVMContext &C = M.getContext(); 
  AttributeList AL;
  // TODO: Set appropriate function attributes.
  FunctionType *FTy = FunctionType::get(
      Type::getInt32Ty(C),                            // returns int
      { Type::getInt32Ty(C),                          // int argc
        PointerType::getUnqual(Type::getInt8PtrTy(C)) // char **argv
      }, false);

  RealmInitRuntime = M.getOrInsertFunction("realmInitRuntime", FTy, AL);
  return RealmInitRuntime;
}

FunctionCallee RealmABI::get_realmFinalize() {
  if(RealmFinalize)
    return RealmFinalize;

  LLVMContext &C = M.getContext(); 
  AttributeList AL;
  // TODO: Set appropriate function attributes.
  FunctionType *FTy = FunctionType::get(Type::getInt8PtrTy(C), {}, false);
  RealmFinalize = M.getOrInsertFunction("realmFinalize", FTy, AL);
  return RealmFinalize;
}

#define REALM_FUNC(name) get_##name()

RealmABI::RealmABI(Module &M) : TapirTarget(M) {
  LLVMContext &C = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  // Initialize any types we need for lowering.
  TaskFuncPtrTy = PointerType::getUnqual(
      FunctionType::get(
           Type::getInt8Ty(C),      // returns void 
           { Type::getInt8PtrTy(C), // const void *args
	     DL.getIntPtrType(C),   // size_t arglen 
	     Type::getInt8PtrTy(C), // const void *user_data
	     DL.getIntPtrType(C),   // size_t user_data_len
	     DL.getIntPtrType(C)    // unsigned long long proc
	   }, false));
}

RealmABI::~RealmABI() {
  //call something that deletes the context struct
}

/// Lower a call to get the grainsize of this Tapir loop.
///
/// The grainsize is computed by the following equation:
///
///     Grainsize = min(2048, ceil(Limit / (8 * workers)))
///
/// This computation is inserted into the preheader of the loop.
Value *RealmABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
  Value *Limit = GrainsizeCall->getArgOperand(0);
  IRBuilder<> Builder(GrainsizeCall);

  // Get 8 * workers
  Value *Workers = Builder.CreateCall(REALM_FUNC(realmGetNumProcs));
  //Value *Workers = Builder.CreateCall(get_realmGetNumProcs()); // no macro
  Value *WorkersX8 = Builder.CreateIntCast(
      Builder.CreateMul(Workers, ConstantInt::get(Workers->getType(), 8)),
      Limit->getType(), false);
  // Compute ceil(limit / 8 * workers) =
  //           (limit + 8 * workers - 1) / (8 * workers)
  Value *SmallLoopVal =
    Builder.CreateUDiv(Builder.CreateSub(Builder.CreateAdd(Limit, WorkersX8),
                                         ConstantInt::get(Limit->getType(), 1)),
                       WorkersX8);
  // Compute min
  Value *LargeLoopVal = ConstantInt::get(Limit->getType(), 2048);
  Value *Cmp = Builder.CreateICmpULT(LargeLoopVal, SmallLoopVal);
  Value *Grainsize = Builder.CreateSelect(Cmp, LargeLoopVal, SmallLoopVal);

  // Replace uses of grainsize intrinsic call with this grainsize value.
  GrainsizeCall->replaceAllUsesWith(Grainsize);
  return Grainsize;
}

Value *RealmABI::getOrCreateBarrier(Value *SyncRegion, Function *F) {
  LLVMContext &C = M.getContext();
  Value* barrier;
  if((barrier = SyncRegionToBarrier[SyncRegion]))
    return barrier;
  else {
    barrier = CallInst::Create(get_createRealmBarrier(), {}, "",
                            F->getEntryBlock().getTerminator());
    SyncRegionToBarrier[SyncRegion] = barrier;

    // Make sure we destroy the barrier at all exit points to prevent memory leaks
    for(BasicBlock &BB : *F) {
      if(isa<ReturnInst>(BB.getTerminator())){
        CallInst::Create(get_destroyRealmBarrier(), {barrier}, "",
                         BB.getTerminator());
      }
    }

    return barrier;
  }
}

void RealmABI::lowerSync(SyncInst &SI) {
  IRBuilder<> builder(&SI); 
  auto F = SI.getParent()->getParent(); 
  auto& C = M.getContext(); 
  Value* SR = SI.getSyncRegion(); 
  auto barrier = getOrCreateBarrier(SR, F); 
  std::vector<Value *> args = {barrier}; 
  builder.CreateCall(get_realmSync(), args);

  BranchInst *PostSync = BranchInst::Create(SI.getSuccessor(0));
  ReplaceInstWithInst(&SI, PostSync);
  return;
}

// Adds entry basic blocks to body of extracted, replacing extracted, and adds
// necessary code to call, i.e. storing arguments in struct
Function* RealmABI::formatFunctionToRealmF(Function* extracted, Instruction* ical){
  std::vector<Value*> LoadedCapturedArgs;
  CallInst *cal = dyn_cast<CallInst>(ical);

  for(auto& a:cal->arg_operands()) {
    LoadedCapturedArgs.push_back(a);
  }

  Module *M = extracted->getParent(); 
  auto& C = M->getContext(); 
  DataLayout DL(M);
  IRBuilder<> CallerIRBuilder(cal);

  //get the argument types
  auto FnParams = extracted->getFunctionType()->params();
  StructType *ArgsTy = StructType::create(FnParams, "anon");
  auto *ArgsPtrTy = PointerType::getUnqual(ArgsTy);

  //Create the canonical TaskFuncPtr
  ArrayRef<Type*> typeArray = {Type::getInt8PtrTy(C), Type::getInt64Ty(C), Type::getInt8PtrTy(C), Type::getInt64Ty(C), Type::getInt64Ty(C)}; //trying int64 as stand-in for Realm::Processor because a ::realm_id_t is ultimately an unsigned long long

  FunctionType *OutlinedFnTy = FunctionType::get(
      Type::getInt8Ty(C), 
      typeArray,
      false);

  Function *OutlinedFn = Function::Create(
      OutlinedFnTy, GlobalValue::InternalLinkage, ".realm_outlined.", M);
  OutlinedFn->addFnAttr(Attribute::AlwaysInline);
  OutlinedFn->addFnAttr(Attribute::NoUnwind);
  OutlinedFn->addFnAttr(Attribute::UWTable);

  //StringRef ArgNames[] = {".args"};
  std::vector<Value*> out_args;
  for (auto &Arg : OutlinedFn->args()) {
    //Arg.setName(ArgNames[out_args.size()]);
    Arg.setName("");
    out_args.push_back(&Arg);
  }

  // Entry Code
  auto *EntryBB = BasicBlock::Create(C, "entry", OutlinedFn, nullptr);
  IRBuilder<> EntryBuilder(EntryBB);
  auto argStructPtr = EntryBuilder.CreateBitCast(out_args[0], ArgsPtrTy); 
  ValueToValueMapTy valmap;

  unsigned int argc = 0;
  for (auto& arg : extracted->args()) {
    auto *DataAddrEP = EntryBuilder.CreateStructGEP(ArgsTy, argStructPtr, argc); 
    auto *DataAddr = EntryBuilder.CreateAlignedLoad(
        DataAddrEP,
        DL.getTypeAllocSize(DataAddrEP->getType()->getPointerElementType()));
    valmap.insert(std::pair<Value*,Value*>(&arg,DataAddr));
    argc++;
  }

  // Replace return values with return zero 
  SmallVector< ReturnInst *,5> retinsts;
  CloneFunctionInto(OutlinedFn, extracted, valmap, true, retinsts);
  EntryBuilder.CreateBr(OutlinedFn->getBasicBlockList().getNextNode(*EntryBB));

  for (auto& ret : retinsts) {
    auto retzero = ReturnInst::Create(C, ConstantInt::get(Type::getInt8Ty(C), 0));
    ReplaceInstWithInst(ret, retzero);
  }

  // Caller code
  auto callerArgStruct = CallerIRBuilder.CreateAlloca(ArgsTy); 

  unsigned int cArgc = 0;
  for (auto& arg : LoadedCapturedArgs) {
    auto *DataAddrEP2 = CallerIRBuilder.CreateStructGEP(ArgsTy, callerArgStruct, cArgc); 
    CallerIRBuilder.CreateAlignedStore(
        LoadedCapturedArgs[cArgc], DataAddrEP2,
        DL.getTypeAllocSize(arg->getType()));
    cArgc++;
  }

  assert(argc == cArgc && "Wrong number of arguments passed to outlined function"); 

  auto outlinedFnPtr = CallerIRBuilder.CreatePointerBitCastOrAddrSpaceCast(
									   OutlinedFn, TaskFuncPtrTy);
  auto argSize = ConstantInt::get(Type::getInt64Ty(C), ArgsTy->getNumElements()); 
  auto argDataSize = ConstantInt::get(Type::getInt64Ty(C), DL.getTypeAllocSize(ArgsTy)); 
  auto argsStructVoidPtr = CallerIRBuilder.CreateBitCast(callerArgStruct, Type::getInt8PtrTy(C)); 

  ArrayRef<Value *> callerArgs = { outlinedFnPtr, argsStructVoidPtr, argSize, barrier }; 

  CallerIRBuilder.CreateCall(get_realmSpawn(), callerArgs); 

  cal->eraseFromParent();
  extracted->eraseFromParent();

  LLVM_DEBUG(OutlinedFn->dump()); 

  return OutlinedFn; 
}

void RealmABI::processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) {
  Function *Outlined = TOI.Outline;
  Instruction *ReplStart = TOI.ReplStart;
  CallBase *ReplCall = cast<CallBase>(TOI.ReplCall);
  BasicBlock *CallBlock = ReplStart->getParent();

  LLVMContext &C = M.getContext();
  const DataLayout &DL = M.getDataLayout();

  Function *OutlinedFnPtr = formatFunctionToRealmF(Outlined, ReplStart);

#if 0
  // At this point, we have a call in the parent to a function containing the
  // task body.  That function takes as its argument a pointer to a structure
  // containing the inputs to the task body.  This structure is initialized in
  // the parent immediately before the call.

  // To match the Realm ABI, we replace the existing call with a call to
  // realmSync from the kitsune-rt realm wrapper.
  IRBuilder<> CallerIRBuilder(ReplCall);
  Value *OutlinedFnPtr = CallerIRBuilder.CreatePointerBitCastOrAddrSpaceCast(
      Outlined, TaskFuncPtrTy);
  AllocaInst *CallerArgStruct = cast<AllocaInst>(ReplCall->getArgOperand(0));
  Type *ArgsTy = CallerArgStruct->getAllocatedType();

  ConstantInt *ArgSize = ConstantInt::get(DL.getIntPtrType(C),
                                          DL.getTypeAllocSize(ArgsTy));
  ConstantInt *ArgNum = ConstantInt::get(DL.getIntPtrType(C),
    					 ArgsTy->getNumContainedTypes());
  CallInst *Call = CallerIRBuilder.CreateCall(
      REALM_FUNC(realmSpawn), { OutlinedFnPtr, 
	                        CallerArgStruct,
                                ArgNum,
	                        CallerArgStruct,
	                        ArgSize});
  Call->setDebugLoc(ReplCall->getDebugLoc());
  TOI.replaceReplCall(Call);
  ReplCall->eraseFromParent();

  // Add lifetime intrinsics for the argument struct.  TODO: Move this logic
  // into underlying LoweringUtils routines?
  CallerIRBuilder.SetInsertPoint(ReplStart);
  CallerIRBuilder.CreateLifetimeStart(CallerArgStruct, ArgSize);
  CallerIRBuilder.SetInsertPoint(CallBlock, ++Call->getIterator());
  CallerIRBuilder.CreateLifetimeEnd(CallerArgStruct, ArgSize);

  if (TOI.ReplUnwind)
    // We assume that realmSpawn dealt with the exception.  But
    // replacing the invocation of the helper function with the call to
    // realmSpawn will remove the terminator from CallBlock.  Restore
    // that terminator here.
    BranchInst::Create(TOI.ReplRet, CallBlock);

  // VERIFY: If we're using realmSpawn, we don't need a separate helper
  // function to manage the allocation of the argument structure.
#endif
}

void RealmABI::preProcessFunction(Function &F, TaskInfo &TI,
				  bool OutliningTapirLoops) {

  if (OutliningTapirLoops)
    // Don't do any preprocessing when outlining Tapir loops.
    return;

  for (Task *T : post_order(TI.getRootTask())) {
    if (T->isRootTask())
      continue;
    DetachInst *Detach = T->getDetach();
    BasicBlock *detB = Detach->getParent();
    BasicBlock *Spawned = T->getEntry();

    // Add a submit to end of task body
    IRBuilder<> footerB(Spawned->getTerminator());
    std::vector<Value*> submitArgs; // realmSync takes no args
    //footerB.CreateCall(get_realmSync(), submitArgs);
  }
}

void RealmABI::postProcessFunction(Function &F, bool OutliningTapirLoops) {
  if (OutliningTapirLoops)
    // Don't do any postprocessing when outlining Tapir loops.
    return;

  //if (F.getName() == "main")
  Module *M = F.getParent();
  LLVMContext &C = M->getContext();
  IRBuilder<> builder(F.getEntryBlock().getFirstNonPHIOrDbg());

  //default values of 1 and nullptr
  //TODO: handle the case where main actually has an argc and argv
  Value* one = ConstantInt::get(Type::getInt32Ty(C), 1);
  Value* null = Constant::getNullValue(PointerType::getUnqual(Type::getInt8PtrTy(C)));

  ArrayRef<Value*> initArgs = {one, null};

  builder.CreateCall(REALM_FUNC(realmInitRuntime), initArgs);
  return;
}

void RealmABI::postProcessHelper(Function &F) {}


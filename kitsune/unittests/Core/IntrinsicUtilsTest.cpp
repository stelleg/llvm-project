//==- IntrinsicUtilsTest.cpp - Unit tests for Kitsune intrinsic utilities --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/IntrinsicUtils.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitIntrinsicUtils, isKitsuneIntrinsic) {
  // This is not a comprehensive list, it just attempts to ensure some sanity.
  EXPECT_TRUE(isKitsuneIntrinsic(Intrinsic::kit_async_launch_kernel));
  EXPECT_TRUE(isKitsuneIntrinsic(Intrinsic::kit_async_prefetch_dtoh));
  EXPECT_TRUE(isKitsuneIntrinsic(Intrinsic::kit_async_prefetch_htod));
  EXPECT_TRUE(isKitsuneIntrinsic(Intrinsic::kit_enable_verbose));
  EXPECT_TRUE(isKitsuneIntrinsic(Intrinsic::kit_memcpy_dtoh));
  EXPECT_TRUE(isKitsuneIntrinsic(Intrinsic::kit_memcpy_htod));

  EXPECT_FALSE(isKitsuneIntrinsic(Intrinsic::prefetch));
  EXPECT_FALSE(isKitsuneIntrinsic(Intrinsic::memcpy));
  EXPECT_FALSE(isKitsuneIntrinsic(Intrinsic::memset));
}

TEST(KitIntrinsicUtils, isKitsuneIntrinsicAsync) {
  // This ought to be a comprehensive list.
  EXPECT_TRUE(isKitsuneIntrinsicAsync(Intrinsic::kit_async_launch_kernel));
  EXPECT_TRUE(isKitsuneIntrinsicAsync(Intrinsic::kit_async_memcpy_dtoh));
  EXPECT_TRUE(isKitsuneIntrinsicAsync(Intrinsic::kit_async_memcpy_htod));
  EXPECT_TRUE(isKitsuneIntrinsicAsync(Intrinsic::kit_async_prefetch_dtoh));
  EXPECT_TRUE(isKitsuneIntrinsicAsync(Intrinsic::kit_async_prefetch_htod));

  EXPECT_FALSE(isKitsuneIntrinsicAsync(Intrinsic::kit_memcpy_dtoh));
  EXPECT_FALSE(isKitsuneIntrinsicAsync(Intrinsic::kit_memcpy_htod));
  EXPECT_FALSE(isKitsuneIntrinsicAsync(Intrinsic::kit_enable_verbose));
  EXPECT_FALSE(isKitsuneIntrinsicAsync(Intrinsic::kit_finalize));
  EXPECT_FALSE(isKitsuneIntrinsicAsync(Intrinsic::kit_initialize));
}

TEST(KitIntrinsicUtils, isKitsuneIntrinsicBlocking) {
  // Most Kitsune intrinsics are blocking. This is mainly here to ensure that
  // this function does not report async calls as blocking. At least the async
  // calls here ought to be comprehensive.
  EXPECT_FALSE(isKitsuneIntrinsicBlocking(Intrinsic::kit_async_launch_kernel));
  EXPECT_FALSE(isKitsuneIntrinsicBlocking(Intrinsic::kit_async_memcpy_dtoh));
  EXPECT_FALSE(isKitsuneIntrinsicBlocking(Intrinsic::kit_async_memcpy_htod));
  EXPECT_FALSE(isKitsuneIntrinsicBlocking(Intrinsic::kit_async_prefetch_dtoh));
  EXPECT_FALSE(isKitsuneIntrinsicBlocking(Intrinsic::kit_async_prefetch_htod));

  EXPECT_TRUE(isKitsuneIntrinsicBlocking(Intrinsic::kit_memcpy_dtoh));
  EXPECT_TRUE(isKitsuneIntrinsicBlocking(Intrinsic::kit_memcpy_htod));
  EXPECT_TRUE(isKitsuneIntrinsicBlocking(Intrinsic::kit_enable_verbose));
  EXPECT_TRUE(isKitsuneIntrinsicBlocking(Intrinsic::kit_finalize));
  EXPECT_TRUE(isKitsuneIntrinsicBlocking(Intrinsic::kit_initialize));
}

TEST(KitIntrinsicUtils, getKernelArgumentsFromLaunch) {
  LLVMContext ctx;
  Module m("", ctx);
  Function *f =
      Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kit_async_launch_kernel);

  FunctionType *fty = f->getFunctionType();
  PointerType *ptr = PointerType::getUnqual(ctx);
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);
  Type *f32 = Type::getFloatTy(ctx);

  Constant *g = m.getOrInsertGlobal("g", ptr);
  Constant *cnull = ConstantPointerNull::get(ptr);
  Constant *c0_4 = ConstantInt::get(i32, 0);
  Constant *c0_8 = ConstantInt::get(i64, 0);
  Constant *ctt = ConstantInt::get(i32, unsigned(TTID::Cuda));
  Constant *cfp = ConstantFP::get(f32, 3.14159);

  CallInst *call0 =
      CallInst::Create(fty, f, {ctt, cnull, cnull, c0_8, c0_4, cnull, cnull});
  CallInst *call1 = CallInst::Create(
      fty, f, {ctt, cnull, cnull, c0_8, c0_4, cnull, cnull, g});
  CallInst *call2 = CallInst::Create(
      fty, f, {ctt, cnull, cnull, c0_8, c0_4, cnull, cnull, g, cfp});
  CallInst *call3 = CallInst::Create(
      fty, f, {ctt, cnull, cnull, c0_8, c0_4, cnull, cnull, g, cfp, ctt});

  EXPECT_EQ(getKernelArgumentsFromLaunch(*call0), std::vector<Value *>());
  EXPECT_EQ(getKernelArgumentsFromLaunch(*call1), std::vector<Value *>({g}));
  EXPECT_EQ(getKernelArgumentsFromLaunch(*call2),
            std::vector<Value *>({g, cfp}));
  EXPECT_EQ(getKernelArgumentsFromLaunch(*call3),
            std::vector<Value *>({g, cfp, ctt}));

  call0->deleteValue();
  call1->deleteValue();
  call2->deleteValue();
  call3->deleteValue();
}

TEST(KitIntrinsicUtils, getStreamFromLaunch) {
  LLVMContext ctx;
  Module m("", ctx);
  Function *f =
      Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kit_async_launch_kernel);

  FunctionType *fty = f->getFunctionType();
  PointerType *ptr = PointerType::getUnqual(ctx);
  Type *i32 = Type::getInt32Ty(ctx);
  Type *i64 = Type::getInt64Ty(ctx);
  Type *f32 = Type::getFloatTy(ctx);

  Constant *g = m.getOrInsertGlobal("g", ptr);
  Constant *cnull = ConstantPointerNull::get(ptr);
  Constant *c0_4 = ConstantInt::get(i32, 0);
  Constant *c0_8 = ConstantInt::get(i64, 0);
  Constant *ctt = ConstantInt::get(i32, unsigned(TTID::Cuda));
  Constant *cfp = ConstantFP::get(f32, 3.14159);

  CallInst *call0 =
      CallInst::Create(fty, f, {ctt, cnull, cnull, c0_8, c0_4, cnull, g});
  CallInst *call1 =
      CallInst::Create(fty, f, {ctt, cnull, cnull, c0_8, c0_4, cnull, g, cfp});

  EXPECT_EQ(getStreamFromLaunch(*call0), g);
  EXPECT_EQ(getStreamFromLaunch(*call1), g);

  call0->deleteValue();
  call1->deleteValue();
}

} // namespace

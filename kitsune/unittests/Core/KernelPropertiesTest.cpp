//===- KernelPropertiesTest.cpp - Unit tests for kernel properties --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/KernelProperties.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KitsuneUtils, createKernelPropertiesGlobal) {
  LLVMContext ctx;
  Module m("", ctx);

  GlobalVariable *g = createKernelPropertiesGlobal("kern_132", TTID::Serial, m);
  g->setName("g0");

  EXPECT_TRUE(m.getGlobalVariable("g0", true));
  EXPECT_TRUE(g->hasInitializer());
  EXPECT_TRUE(isa<ConstantAggregateZero>(g->getInitializer()));
  EXPECT_TRUE(g->hasAttribute("kit_kernel_props"));
  EXPECT_EQ(g->getAttribute("kit_kernel_props").getValueAsString(), "kern_132");
}

} // namespace

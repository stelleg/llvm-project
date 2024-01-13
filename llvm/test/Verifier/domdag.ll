; RUN: not llvm-as < %s 2>&1 | FileCheck %s

define i32 @test(i32 %i, i32 %j, i1 %c) {
	br i1 %c, label %A, label %B
A:
  %a = mul i32 %i, 2 
	br i1 %c, label %B, label %C
B:
  %b = add i32 %j, 2
	br i1 %c, label %C, label %A
C:
	%x = add i32 %a, %b             
	ret i32 %x
}

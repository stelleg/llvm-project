export FILEBASE=$1

export BIN=/home/shevitz/kitsune/kitsune/build/bin
export CLANG=${BIN}/clang++
export OPT=${BIN}/opt
export LLC=${BIN}/llc
export CLANGCHECK=${BIN}/check-clang

#############################################
# Clang options
#############################################

# no optnone but still has noinline
CLANG_FLAGS="-ftapir=none -fno-exceptions -O0 -Xclang -disable-O0-optnone -std=c++17"

# no optnone but still has noinline
#CLANG_FLAGS="-fforall -fno-exceptions -O0 -Xclang -disable-O0-optnone -std=c++17"

# serial codegen (no Tapir)
#CLANG_FLAGS="-fno-exceptions -O0 -Xclang -disable-O0-optnone"

# prevent noinline in the IR, but no transforms
#CLANG_FLAGS="-fforall -fno-exceptions -fno-vectorize -O1 -mllvm -disable-llvm-optzns"

# opt O1 without exceptions
#CLANG_FLAGS="-fforall -fno-exceptions -O1 -fno-vectorize"

# no optnone, allow exceptions
#CLANG_FLAGS="-O0 -Xclang -disable-O0-optnone"

#############################################
# opt options
#############################################

#OPT_FLAGS="-mem2reg"
#OPT_FLAGS="-O2"
#OPT_FLAGS="-inline"
OPT_FLAGS=

#############################################
# file names
#############################################

export SOURCE_FILENAME=${FILEBASE}.cpp
export IR_FILENAME=${FILEBASE}.ll
export OPT_FILENAME=${FILEBASE}_opt.ll
export BACKEND_FILENAME=${FILEBASE}_backend.ll
export ASSEMBLY_FILENAME=${FILEBASE}.s
export EXECUTABLE_FILENAME=${FILEBASE}

#############################################
# execute stuff
#############################################

# verbose
set -x

# generate the executable via all intermediate steps with optimizations
<<COMMENT
COMMENT

# remove existing files (ignore if not existing)
rm -f  ${IR_FILENAME} ${OPT_FILENAME} ${BACKEND_FILENAME} ${ASSEMBLY_FILENAME} ${EXECUTABLE_FILENAME}

# compile to IR
${CLANG} -S -emit-llvm ${CLANG_FLAGS} ${SOURCE_FILENAME} -o ${IR_FILENAME}

# run the optimizer if the optimizer flags are not null
if [[ ! -z "$OPT_FLAGS" ]] ; 
	then \
		${OPT} -S ${OPT_FLAGS} ${IR_FILENAME} -o ${OPT_FILENAME} && \

		# lower the tapir to qthreads
		${OPT} -S -tapir-target=qthreads -tapir2target ${OPT_FILENAME} -o ${BACKEND_FILENAME};

	else
		# lower the tapir to qthreads
		${OPT} -S -tapir-target=qthreads -tapir2target ${IR_FILENAME} -o ${BACKEND_FILENAME};
fi

# run the static compiler to generate the assembly_file
${LLC} ${BACKEND_FILENAME} -o ${ASSEMBLY_FILENAME}

# link the executable
${CLANG} -ftapir=qthreads ${ASSEMBLY_FILENAME} -o ${EXECUTABLE_FILENAME}






<<COMMENT
# generate the executable starting from IR
rm -f ${BACKEND_FILENAME} ${ASSEMBLY_FILENAME} ${EXECUTABLE_FILENAME}
${OPT} -S -tapir-target=qthreads -tapir2target ${IR_FILENAME} -o ${BACKEND_FILENAME};
${LLC} ${BACKEND_FILENAME} -o ${ASSEMBLY_FILENAME}
${CLANG} -ftapir=qthreads ${ASSEMBLY_FILENAME} -o ${EXECUTABLE_FILENAME}
COMMENT



<<COMMENT
# this is a cheat sheet for doing simple clang stuff

# run gdb
gdb -ex "set follow-fork-mode child" --args ${CLANG} -S -emit-llvm ${CLANG_FLAGS} ${SOURCE_FILENAME} -o ${IR_FILENAME}

# simply compile and run a program
clang++ test_forall.cpp && ./a.out

# just generate the IR
clang++ -S -emit-llvm test_forall.cpp

# generate the .dot file of the CFG
opt -dot-cfg test_forall.ll
opt -dot-cfg ${FILEBASE}${OPT_FLAGS}.ll

# generate the pdf of the CFG
dot -Tpdf -o test_forall.pdf .main.dot -Gsize=8.5,11\!
dot -Tpdf -o ${FILEBASE}${OPT_FLAGS}.pdf .main.dot -Gsize=8.5,11\!

# go from .ll to an executable
llc ${FILEBASE}.ll && clang ${FILEBASE}.s && ./a.out

llc hacked.ll && clang hacked.s && ./a.out

# find out what passes run
${OPT} -S -debug-pass=Arguments ${OPT_FLAGS} ${FILEBASE}.ll
llvm-as </dev/null | opt -O1 -disable-output -debug-pass=Arguments

# generate the executables all in one
<<COMMENT

# generate IR
${CLANG} -S -emit-llvm ${BUILD_FLAGS} ${FILEBASE}.cpp -o ${FILEBASE}.ll

# generate executable (reverses the thread order)
${CLANG} ${BUILD_FLAGS} ${OPTIMIZATION_FLAGS} -ftapir=qthreads ${FILEBASE}.cpp -o ${FILEBASE}_qthreads

# one line build
${CLANG} -fforall -ftapir=qthreads test_forall.cpp -o test_forall_qthreads
${CLANG} -fforall -S -emit-llvm test_forall.cpp -o test_forall.ll

## generate executable (serial executable, since no ftapir)
## ${CLANG} ${BUILD_FLAGS} ${OPTIMIZATION_FLAGS} ${FILEBASE}.cpp -o ${FILEBASE}_qthreads

# dump AST
${CLANG} -Xclang -ast-dump ${FILEBASE}.cpp 
${CLANGCHECK} ${FILEBASE}.cpp -ast-dump -ast-dump-filter main --

COMMENT



<<COMMENT
# obsolete one line compile( if clang is built with cmake flags )
#${CLANG} ${FORALL} -${OPTIMIZATION_FLAGS} -L/projects/kitsune/qthreads/lib -ftapir=qthreads ${FILEBASE}.cpp -o ${FILEBASE}_qthreads -lqthread -lpthread
COMMENT

set +x


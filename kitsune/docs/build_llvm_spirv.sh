# This script describes and implements the building of LLVM+Tapir -> SPIRV and the MESA OpenCL runtime library

# Set up env
mkdir kitsuneopencl; cd kitsuneopencl; rootdir=$PWD
installPrefix=$HOME/usr/`uname -m`
module load clang/11.0.1

# Get the sources
#git clone git@github.com:lanl/kitsune; cd kitsune; git checkout opencl/11.x; cd $rootdir
#git clone git@github.com:khronosgroup/spirv-llvm-translator; cd spirv-llvm-translator; git checkout llvm_release_110; cd $rootdir
#git clone https://gitlab.freedesktop.org/mesa/mesa.git; cd mesa; git checkout 21.1; cd $rootdir

# First, we build the spirv tool against an existing clang install
mkdir spirvbuild; cd spirvbuild; cmake $rootdir/spirv-llvm-translator \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$installPrefix \
	-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
	-DCMAKE_SKIP_RPATH=ON \
	-DLLVM_EXTERNAL_LIT=lit \
  -Wno-dev \
	-G Ninja
ninja install
cd $rootdir

# Next, configure and build llvm without the translator
mkdir kitsunebuild; cd kitsunebuild; cmake $rootdir/kitsune/llvm \
	-DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;libc;libclc;libcxx;libcxxabi;lld;mlir;polly;kitsune" \
	-DCMAKE_INSTALL_PREFIX=$installPrefix \
	-DLLVM_ENABLE_RTTI=ON \
	-DCMAKE_BUILD_TYPE=Release \
  -DLLVM_LINK_LLVM_DYLIB=ON \
	-DCLANG_LINK_CLANG_DYLIB=ON \
	-DLLVM_INSTALL_UTILS=ON \
	-G Ninja 
ninja install
cd $rootdir

# Now, we configure and build mesa+clover
cd mesa; meson $rootdir/mesabuild \
	-Dplatforms=x11 \
  -Dvulkan-drivers=amd \
  -Dgallium-drivers=radeonsi \
	-Dgallium-opencl=icd \
	-Dopencl-spirv=true \
	-Dprefix="~/usr/x86_64" \
	-Dbuildtype=release \
	-Dbackend=ninja
cd $rootdir/mesabuild; meson compile; meson test; meson install


## Using Kitsune+Tapir 

As the name suggestions there are two components to the Kitsune+Tapir toolchain.  

First, Kitsune provides a set of extensions to Clang that support the front-end components of compiling parallel constructs.  In this case, there are two primary paths to use these language features: 

  * Use of the ``forall`` keyword.  At the level of the syntatic rules a ``forall`` keyword are identical to C/C++ ``for`` loops.  However, the parallel semantics of the loop introduce new restrictions that developers should be aware of:

     * The loop body will execute in parallel where each iteration through the loop should be independent of all others.  Developers should also make no assumptions about any assumed ordering of the interations in the parallel form of execution.  In a nutshell, these restrictions are no different any many other parallel loop constructs.

  * The parallelization of *some* forms of the [Kokkos]() C++ ``parallel_for`` constructs. This support is enabled by using the ``-fkokkos`` command line argument.  The same rules for parallel loop execution follow both Kokkos and the ``forall`` semantics.  Note that only the lambda form of ``parallel_for`` is recognized by Kitsune.  There are several examples of using Kokkos under the ``kitsune`` directory at the top-level of the source code repository (``kitsune/examples/kokkos``). 

     * Some limited forms of ``Kokkos::parallel_for`` using ``MDRange`` are also supported.  The examples provide a set of currently supported constructs. 

     * ``Kokkos::parallel_reduce`` support is currently under development. **(TODO: This needs to be updated when reductions are completed.)**

**(TODO: Document other language-level constructs -- e.g., ``spawn``, ``sync``.)**

The second component of the complilation stage is Tapir that is implemented as a series of extensions to LLVM.  Tapir is responsible for taking the parallel represetnations of code provided by Kitsune and (1) optimizing them and (2) taking the optimized code and transforming it into a parallel for for a specific architecture and corresponding runtime ABI target.  The architecture target for the code is primarily defined by the runtime target.  For example: 

  * `-ftapir=serial`: will transform the parallel intermediate form used by Tapir into a serial CPU code. 
  * `-ftapir=opencilk`: will transform the parallel intermediate form used by Tapir into a CPU executable that leverages the OpenCilk runtime system for parallelism. 
  * `-ftapir=openmp`: will transform the parallel intermediate form used by Tapir into a CPU executable that leverages the OpenMP runtime system (even if the input source program is not using OpenMP).
  * `-ftapir=cudatk`: will transform the parallel intermediate form used by Tapir into a runtime target for supporting CUDA and NVIDIA's GPU architectures. `cudatk` is shorthand for a CUDA Toolkit runtime that is part of the Kitsune code base and simplifies some of the details of code generation for CUDA. **(TODO: This transform is still under development and should not be considered robust.)
  * `-ftapir=hip`: will transform the parallel intermediate form used by Tapir into a runtime target for supporting HIP and AMD's GPU architectures.  Like the CUDA target above, there is a HIP-specific runtime library that is packaged with Kitsune that simplifies some aspects of code generation for the AMD software stack and GPU hardware. 
  * `-ftapir=realm`: will tranform the parallel intermediate form used by Tapir into a runtime target that supports the Realm runtime system that is used the low-level runtime system used by the Legion Programming System. 
  * `-ftapir=qthreads`: will tranform the parallel intermediate form used by Tapir into a runtime target that supports CPU execution using the Qthreads runtime system. 

  **(TODO: Flush out the details for other targets -- e.g. OpenCL.)** 

Based on the selected runtime and architecture target, the result executable each have their own unique aspects for supporting parallel execution parameters.  Many are controlled by enviornment variables and are dependent upon the runtime system.  A quick overview of some of these parameters are quickly discussed below. 

**OpenCilk Runtime Target**: The OpenCilk runtime target supports one primary enviornment variable that controls the number of worker threads that will be used to execute supported language constructs (e.g., `forall`).  This enviornment variable is `CILK_NWORKERS`:
```bash
$ clang++ -ftapir=opencilk ... my_program.cpp 
$ export CILK_NWORKERS=16   # use 16 worker threads during execution. 
$ a.out 
``` 

**QThreads Runtime Target**: The Qthreads runtime has [several settings](https://cs.sandia.gov/qthreads/man/qthread_init.html#toc3) via the environment that can impact behavior and 
performance.  At a minimum setting `QTHREAD_NUM_SHEPHERDS` will allow you to control the number of threads assigned to the execution of an executable. 
 ```bash 
 $ clang++ -ftapir=qthreads ... file.cpp 
 $ export QTHREAD_NUMBER_SHEPHERDS=16 # use 16 threads during execution. 
 $ a.out
 ``` 

**Realm Runtime Target**: When running a Realm target you can provide a full command 
line via the `REALM_DEFAULT_ARGS` enviornment variable. More details on the various 
command line arguments supported by Realm can be found [here](https://legion.stanford.edu/starting/); look for the *Command-Line Flags* section. 

```bash 
$ clang++ -ftapir=realm ... file.cpp 
$ export REALM_DEFAULT_ARGS="-ll:cpu 1 -ll:force_kthreads -level task=2,taskreg=2"
$ a.out 
``` 


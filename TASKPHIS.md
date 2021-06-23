Right now we can generate Tapir IR with Phi nodes, e.g. l2 norm:

    int triangular(uint64_t n){
      uint64_t sum = 0; 
      cilk_for(uint64_t i=1; i<n; i++){
        sum += i; 
      }
      return sum;
    }

If we run `clang -ftapir=none -O1 -S -emit-llvm` on this code (-O1
will ensure mem2reg is run, but not other passes, such as strip
mining, which will still break), we'll get tapir code that has a phi
node for sum instead of just reads and writes to an alloca. The next
steps are to figure out what needs to change (likely SCEV, as TB
suggested) to apply the kinds of recurrence optimizations that apply
to sequential code.

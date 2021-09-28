#include<kitsune.h>
#include<gpu.h>
#include<assert.h>

int main(int argc, char** argv){
  int n = argc > 1 ? atoi(argv[1]) : 2048; 
  float* A = (float*) gpuManagedMalloc(n * sizeof(float)); 
  float* B = (float*) gpuManagedMalloc(n * sizeof(float)); 
  float* C = (float*) gpuManagedMalloc(n * sizeof(float)); 
  for(int i=0; i<n; i++){
    A[i] = 3.14;
    B[i] = 2.71; 
  }
  forall(int i=0; i<n; i++){
    C[i] = A[i] + B[i];  
  }
  for(int i=0; i<n; i++){
    if(C[i] != A[i] + B[i]){
      printf("Failure\n");
      exit(i);
    }
  }
  printf("Success\n"); 
}


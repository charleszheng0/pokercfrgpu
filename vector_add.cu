%%writefile vector_add.cu
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd (float* A, float* B, float* C, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i<N){
    C[i] = A[i] + B[i];
  }
}

int main(){
  const int N = 1000;
  const size_t bytes = N * sizeof(float);
  float A[N], B[N], C[N]; 

  for (int i = 0; i < N; i++) {
      A[i] = i; // A = [0, 1, 2, 3...]
      B[i] = i * 2; // B = [0, 2, 4...]
  }
float *d_A, *d_B, *d_C; 
cudaMalloc((void**)&d_A, bytes); 
cudaMalloc((void**)&d_B, bytes); 
cudaMalloc((void**)&d_C, bytes); // void** is *d_A pointing to another pointer address
cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice); // changing the actual  array in gpu
cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);


int threadsPerBlock = 256; 
int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock; 

vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N); 

cudaError_t err = cudaGetLastError(); 
if (err != cudaSuccess){
  printf("Kernel launch failed: %s\n", cudaGetErrorString(err)); 
  return 1; 
}
err = cudaDeviceSynchronize();
if (err != cudaSuccess) {
    printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
    return 1;
}

cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

for (int i = 0; i< 100; i++){
  printf("%f + %f = %f\n", A[i], B[i], C[i]);
}
cudaFree(d_A); 
cudaFree(d_B); 
cudaFree(d_C); 

}

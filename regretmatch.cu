#include <stdio.h>
#include <cuda_runtime.h>
//GPU
// regrets = flat array containing two regrets per infoset
// strategy = flat output array containing two probabilities per infoset
// numInfosets = number of information sets (12)

__global__ void regretMatching(
    const float* regrets,
    float* strategy,
    int numInfosets
) {
    // TODO 1: Calculate global thread index i
    // blockIdx.x = current block
    // blockDim.x = threads per block
    // threadIdx.x = current thread inside its block
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO 2: Bounds check
    // Only process if i < numInfosets
    if (i<numInfosets){

    

    // TODO 3: Read both regrets
    float r0 = regrets[2 * i]
    float r1 = regrets[2 * i + 1]


    // TODO 4: Clamp negative regrets to zero
    float pos0 = fmaxf(0.0f, r0);
    float pos1 = fmaxf(0.0f, r1);
    // Returns the larger of two floats
    // Example: fmaxf(-3.0f, 0.0f) returns 0


    // TODO 5: Calculate sum of positive regrets
    float sum = pos0+pos1;
    float p0, p1; 
    }
    // TODO 6: Calculate action probabilities
    // If sum > 0, divide each positive regret by sum
    // Otherwise, use 0.5 for both actions
    if (sum>0.0f){
        p0 = pos0/sum;
        p1 = pos1/sum; 
    }
    else{
        p0 = 0.5f;
        p1=0.5f;
    }
    // TODO 7: Store the two probabilities
    strategy[2 * i] = p0;
    strategy[2 * i + 1] = p1;
}


// =====================================================
// CPU MAIN
// =====================================================

int main() {

    const int numInfosets = 12;
    const int numActions = 2;

    const size_t bytes =
        numInfosets * numActions * sizeof(float);

    // Example regrets in your Kuhn Poker infoset order
    float regrets[] = {
        -3, 7,      // J
         4, 2,      // Q
        -2, -5,     // K
         0, 0,      // Jc
         0, 0,      // Qc
         0, 0,      // Kc
         0, 0,      // Jb
         0, 0,      // Qb
         0, 0,      // Kb
         0, 0,      // Jcb
         0, 0,      // Qcb
         0, 0       // Kcb
    };

    float strategy[24];

    float *d_regrets, *d_strategy; // declare on cpu

    // TODO 8: Allocate GPU memory
    // cudaMalloc(void** devPtr, size_t size)
    // Allocate bytes for d_regrets and d_strategy
    cudaMalloc((void**)&d_regrets, bytes); // allocate gpu memory and store the addresses to the pointers
    cudaMalloc((void**)&d_strategy, bytes);

    // TODO 9: Copy regrets from CPU to GPU
    // cudaMemcpy(dst, src, count, kind)
    // kind = cudaMemcpyHostToDevice
    cudaMemcpy(d_regrets, regrets, bytes, cudaMemcpyHostToDevice); 

    // TODO 10: Configure threads and blocks
    // Use 256 threads per block
    // blocksPerGrid = ceiling(numInfosets / threadsPerBlock)
    int threadsPerBlock = 256;
    blocksPerGrid = ceiling(numInfosets/threadsPerBlock);

    // TODO 11: Launch regretMatching
    // kernel<<<blocksPerGrid, threadsPerBlock>>>(
    //     d_regrets, d_strategy, numInfosets
    // );
    regretMatching<<<blocksPerGrid, threadsPerBlock>>>(d_regrets, d_strategy, numInfosets);


    // TODO 12: Check kernel errors
    // cudaGetLastError() checks for launch errors
    // cudaDeviceSynchronize() waits for completion
    // and reports execution errors
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
    

    // TODO 13: Copy strategy from GPU to CPU
    // cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost)
    cudaMemcpy(&strategy, d_strategy, sizeof(strategy), cudaMemcpyDeviceToHost);

    // Print strategies for all 12 information sets
    const char* names[] = {
        "J", "Q", "K",
        "Jc", "Qc", "Kc",
        "Jb", "Qb", "Kb",
        "Jcb", "Qcb", "Kcb"
    };

    for (int i = 0; i < numInfosets; i++) {
        printf("%s: [%.3f, %.3f]\n",
               names[i],
               strategy[2 * i],
               strategy[2 * i + 1]);
    }

    // TODO 14: Free GPU memory
    // cudaFree(pointer)
    cudaFree(d_regrets); 
    cudaFree(d_strategy);

    return 0;
}
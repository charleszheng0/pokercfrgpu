%%writefile reduction.cu

#include <stdio.h>
#include <cuda_runtime.h>

// One block reduces an array of N floats.
// For now, N = 256 and we launch exactly one block.

__global__ void reduceSum(const float* input, float* output, int N) {

    // Shared memory accessible to all threads in this block
    __shared__ float temp[256];

    // TODO 1: Get thread index using threadIdx.x
    int i = threadIdx.x;

    // TODO 2: Load input[i] into temp[i]
    temp[i] = input[i]; 
    

    // TODO 3: Synchronize threads
    __syncthreads(); 
    // TODO 4: Repeatedly add pairs of values
    // using a loop that halves the active threads
    for (int stride = blockDim.x/2; stride > 0; stride/=2){
        if (i<stride){
            temp[i] += temp[stride+i];
        }
        __syncthreads(); 
    }


    // TODO 5: Thread 0 writes final sum to output[0]
    if (i ==0){
        output[0] = temp[0];
    }
}

int main() {
    const int N = 256;
    const size_t bytes = N * sizeof(float);

    float input[N];

    for (int i = 0; i < N; i++) {
        input[i] = 1.0f;
    }

    float result = 0.0f;

    float *d_input, *d_output;

    // TODO 6: Allocate GPU memory
    cudaMalloc((void**)&d_input, bytes);
    cudaMalloc((void**)&d_output, bytes);

    // TODO 7: Copy input to GPU
    cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);
    // TODO 8: Launch kernel with 1 block and 256 threads
    reduceSum<<<1, 256>>>(d_input, d_output, N);
    // TODO 9: Copy result back
    cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Sum = %f\n", result);

    // TODO 10: Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
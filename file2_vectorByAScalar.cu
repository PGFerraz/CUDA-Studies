// Creating a kernel that multiplies all vector elements by a scalar passed as a parameter

#include "cuda_runtime.h"
#include "cstdio"

// Kernel
__global__ void mulVectSc(const int *d_v, int *d_a, int N, int s)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N)
    {
        d_a[i] = d_v[i] * s;
    }
}

// Host
int main(void)
{
    // Defining device pointers
    // vector, result
    cudaDeviceReset();
    int *d_v, *d_a;
    
    // Number of elements in the vector
    int N = 1024;
    
    // Scalar that will multiply the vector
    int scalar = 2;

    // Creating Host vectors
    int *h_v = new int[N];
    int *h_a = new int[N];

    // Initializing Host vector values
    for (int i = 0; i < N; i++)
    {
        h_v[i] = ((i + 1) * 2) + 64;
    }

    // Allocating memory on the Device
    cudaMalloc(&d_v, N * sizeof(int));
    cudaMalloc(&d_a, N * sizeof(int));

    // Copying data from Host to Device
    cudaMemcpy(d_v, h_v, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Defining the number of threads per block and blocks per grid
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launching the Kernel
    mulVectSc<<<blocksPerGrid, threadsPerBlock>>>(d_v, d_a, N, scalar);
    cudaDeviceSynchronize();

    // Copying the result back to Host
    cudaMemcpy(h_a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Displaying results
    for(int i=0; i < N; i++)
    {
        printf(" %d" , h_a[i]);
        printf("\n");
    }

    // Freeing memory
    cudaFree(d_v); cudaFree(d_a);
}
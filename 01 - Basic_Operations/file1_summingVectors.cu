// Summing two vectors with paralelism
#include"cuda_runtime.h"
#include"cstdio"

// Kernel
__global__ void vecSum(const int *d_a, const int *d_b, int *d_c, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        d_c[i] = d_a[i] + d_b[i];
    }
}

// Host
int main(void)
{
    cudaDeviceReset();
    int *d_a, *d_b, *d_c;

    // Defining the number of elements in the vectors
    int N = 1024;

    int *h_a = new int[N];
    int *h_b = new int[N];
    int *h_c = new int[N];

    // Initialize first
    for (int i = 0; i < N; i++) {
        h_a[i] = i + 1;
        h_b[i] = i * 2;
    }

    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));

    // Copy afterwards
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vecSum<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    // Copy back to host
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0; i < N; i++)
    {
        printf(" %d" , h_c[i]);
        printf("\n");
    }
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

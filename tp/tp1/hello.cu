#include "cuda.h"
#include <cstdio>


__global__ void cudaHello() {
  printf("Hello from block %d/%d, thread %d/%d\n", blockIdx.x, gridDim.x,
         threadIdx.x, blockDim.x);
}

int main() {
  int numBlocks = 1;
  int blockSize = 64;
  cudaHello<<<numBlocks, blockSize>>>();

  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n",
           cudaGetErrorString(cudaerr));

  const int configs[][2] = {{2, 32}, {4, 16}, {8, 8},
                            {16, 4}, {32, 2}, {64, 1}};

  for (const auto &cfg : configs) {
    numBlocks = cfg[0];
    blockSize = cfg[1];
    cudaHello<<<numBlocks, blockSize>>>();
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
      printf("kernel launch failed with error \"%s\".\n",
             cudaGetErrorString(cudaerr));
      return 1;
    }
  }

  return 0;
}

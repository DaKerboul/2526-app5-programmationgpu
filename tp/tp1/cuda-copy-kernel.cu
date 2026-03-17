#include "cuda.h"
#include <cstdio>
#include <iostream>


using namespace std;

__global__ void cudaCopyByBlocks(float *tab0, const float *tab1, int size) {
  int idx;
  // Compute the correct idx
  idx = blockIdx.x;
  if (idx < size) {
    tab0[idx] = tab1[idx];
  }
}

__global__ void cudaCopyByBlocksThreads(float *tab0, const float *tab1,
                                        int size) {
  int idx;
  // Compute the correct idx in terms of blockIdx.x, threadIdx.x, and blockDim.x
  idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    tab0[idx] = tab1[idx];
  }
}

int main(int argc, char **argv) {
  float *A, *B, *dA, *dB;
  int N, i;

  if (argc < 2) {
    printf("Usage: %s N\n", argv[0]);
    return 0;
  }
  N = atoi(argv[1]);

  // Initialization
  A = (float *)malloc(sizeof(float) * N);
  B = (float *)malloc(sizeof(float) * N);
  for (i = 0; i < N; i++) {
    A[i] = (float)i;
    B[i] = 0.0f;
  }

  // Allocate dynamic arrays dA and dB of size N on the GPU with cudaMalloc
  cudaMalloc(&dA, sizeof(float) * N);
  cudaMalloc(&dB, sizeof(float) * N);

  // Copy A into dA and B into dB
  cudaMemcpy(dA, A, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(float) * N, cudaMemcpyHostToDevice);

  // Copy dA into dB using the kernel cudaCopyByBlocks
  cudaCopyByBlocks<<<N, 1>>>(dB, dA, N);
  // cudaCopyByBlocks<<<...,...>>>(...) ???

  // Wait for kernel cudaCopyByBlocks to finish
  // Attendre que le kernel cudaCopyByBlocks termine
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
    printf("Kernel execution failed with error: \"%s\".\n",
           cudaGetErrorString(cudaerr));
  }

  // Copy dB into B for verification
  // Copier dB dans B pour la verification
  cudaMemcpy(B, dB, sizeof(float) * N, cudaMemcpyDeviceToHost);

  // Verify the results on the CPU by comparing B with A
  // Verifier le resultat en CPU en comparant B avec A
  for (i = 0; i < N; i++) {
    if (A[i] != B[i]) {
      break;
    }
  }
  if (i < N) {
    cout << "La copie est incorrecte!\n";
  } else {
    cout << "La copie est correcte!\n";
  }

  // Reinitialize B to zero, then copy B into dB again to test the second copy
  // kernel Remettre B a zero puis recopier dans dB tester le deuxieme kernel de
  // copie
  for (int i = 0; i < N; i++) {
    B[i] = 0.0f;
  }
  cudaMemcpy(dB, B, sizeof(float) * N, cudaMemcpyHostToDevice);

  // Copy dA into dB with the kernel cudaCopyByBlocksThreads
  // Copier dA dans dB avec le kernel cudaCopyByBlocksThreads
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  cudaCopyByBlocksThreads<<<numBlocks, blockSize>>>(dB, dA, N);
  // cudaCopyByBlocksThreads<<<...,...>>>(...) ???

  // Wait for the kernel cudaCopyByBlocksThreads to finish
  // Attendre que le kernel cudaCopyByBlocksThreads termine
  cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
    printf("L'execution du kernel a echoue avec le code d'erreur \"%s\".\n",
           cudaGetErrorString(cudaerr));
  }

  // Copy dB into B for verification
  // Copier dB dans B pour la verification
  cudaMemcpy(B, dB, sizeof(float) * N, cudaMemcpyDeviceToHost);

  // Verify the results on the CPU by comparing B with A
  // Verifier le resultat en CPU en comparant B avec A
  for (i = 0; i < N; i++) {
    if (A[i] != B[i]) {
      break;
    }
  }
  if (i < N) {
    cout << "La copie est incorrecte!\n";
  } else {
    cout << "La copie est correcte!\n";
  }

  // Deallocate arrays dA[N] and dB[N] on the GPU
  // Desaollouer le tableau dA[N] et dB[N] sur le GPU
  cudaFree(dA);
  cudaFree(dB);

  // Deallocate A and B
  // Desallouer A et B
  free(A);
  free(B);

  return 0;
}

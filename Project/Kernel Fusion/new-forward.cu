#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>

#define TILE_WIDTH 16

// kernel to perform shared memory matrix multiplication
__global__ void matrixMultiplyShared(const float *A, const float *input, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns, int H, int W, int K, int S,int C2)
{
#define in_4d(i3, i2, i1, i0) input[(i3) * (C2 * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int b = blockIdx.z;

  int W_out = (W - K) / S + 1;

  __shared__ __half subTileB[TILE_WIDTH][TILE_WIDTH];
  __shared__ __half subTileA[TILE_WIDTH][TILE_WIDTH];

  // Identify the row and column of the C element to work on
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  __half Pvalue = 0;
  // Loop over the A and B tiles required to compute the C element
  for (int q = 0; q < ((numAColumns + TILE_WIDTH - 1) / TILE_WIDTH); ++q)
  {
    int h_unroll = q * TILE_WIDTH + ty;
    int w_unroll = Col;

    int h = w_unroll / W_out;
    int w = w_unroll % W_out;

    int c = h_unroll / (K * K);

    int transposed_h = h_unroll - c * K * K;

    int p = transposed_h / K;
    int q2 = transposed_h % K;

    int input_h = h * S;
    int input_w = w * S;

    // Collaborative loading of M and N tiles into shared memory
    if (Row < numARows && (q * TILE_WIDTH + tx) < numAColumns)
      subTileA[ty][tx] = __float2half(A[Row * numAColumns + q * TILE_WIDTH + tx]);
    else
      subTileA[ty][tx] = __float2half(0);
    if (Col < numBColumns && (q * TILE_WIDTH + ty) < numBRows)
      subTileB[ty][tx] = __float2half(in_4d(b, c, input_h + p, input_w + q2));
    else
      subTileB[ty][tx] = __float2half(0);
    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; ++k)
      Pvalue += __hmul(subTileA[ty][k], subTileB[k][tx]);
    __syncthreads();
  }
  if (Row < numCRows && Col < numCColumns)
    C[b * numCColumns * numCRows + Row * numCColumns + Col] = __half2float(Pvalue);
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
  // Allocate memory and copy over the relevant data structures to the GPU

  // We pass double pointers for you to initialize the relevant device pointers,
  //  which are passed to the other two functions.

  // Useful snippet for error checking
  // cudaError_t error = cudaGetLastError();
  // if(error != cudaSuccess)
  // {
  //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
  //     exit(-1);
  // }
  const int H_out = (H - K) / S + 1;
  const int W_out = (W - K) / S + 1;

  cudaMalloc(device_output_ptr, B * M * H_out * W_out * sizeof(float));
  cudaMalloc(device_input_ptr, B * C * H * W * sizeof(float));
  cudaMalloc(device_mask_ptr, M * C * K * K * sizeof(float));

  cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
  // Set the kernel dimensions and call the kernel
  const int H_out = (H - K) / S + 1;
  const int W_out = (W - K) / S + 1;

  int numARows = M;
  int numAColumns = C * K * K;
  int numBRows = numAColumns;
  int numBColumns = H_out * W_out;
  int numCRows = M;
  int numCColumns = numBColumns;

  int W_size = (numCColumns + TILE_WIDTH - 1) / TILE_WIDTH;
  int H_size = (numCRows + TILE_WIDTH - 1) / TILE_WIDTH;
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 gridDim(W_size, H_size, B);

  matrixMultiplyShared<<<gridDim, blockDim>>>(device_mask, device_input, device_output, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns, H, W, K, S, C);

  // cudaError_t error = cudaGetLastError();
  // if (error != cudaSuccess)
  // {
  //   std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
  //   exit(-1);
  // }
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
  const int H_out = (H - K) / S + 1;
  const int W_out = (W - K) / S + 1;
  // Copy the output back to host
  cudaMemcpy(host_output, device_output, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);
  // Free device memory
  cudaFree(device_output);
  cudaFree(device_input);
  cudaFree(device_mask);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    exit(-1);
  }
}

__host__ void GPUInterface::get_device_properties()
{
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  for (int dev = 0; dev < deviceCount; dev++)
  {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
    std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
    std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
    std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
    std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, " << deviceProp.maxThreadsDim[1] << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
    std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, " << deviceProp.maxGridSize[1] << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
    std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
  }
}

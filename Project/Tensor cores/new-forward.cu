#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define TILE_WIDTH 16

// kernel to perform shared memory matrix multiplication
__global__ void matrixMultiplyShared(const float *A, const float *input, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns, int H, int W, int K, int S, int C2)
{
#define in_4d(i3, i2, i1, i0) input[(i3) * (C2 * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

  __shared__ __half subTileB[TILE_WIDTH * TILE_WIDTH];
  __shared__ __half subTileA[TILE_WIDTH * TILE_WIDTH];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int b = blockIdx.z;

  int W_out = (W - K) / S + 1;

  wmma::fragment<wmma::matrix_a, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH, float> acc_frag;
  wmma::fragment<wmma::accumulator, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH, float> c_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  // Identify the row and column of the C element to work on
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  int ldc = numCColumns;

  int warpM = (blockIdx.x * blockDim.x + threadIdx.x)/warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
  // Loop over the A and B tiles require-d to compute the C element
  for (int q = 0; q < ((numAColumns + TILE_WIDTH - 1) / TILE_WIDTH); ++q)
  {
    int h_unroll = q + ty;
    int w_unroll = Col;

    int h = w_unroll / W_out;
    int w = w_unroll % W_out;

    int c = h_unroll / (K * K);

    int transposed_h = h_unroll - c * K * K;

    int p = transposed_h / K;
    int q2 = transposed_h % K;

    int input_h = h * S;
    int input_w = w * S;

    int aRow = warpM * TILE_WIDTH;
    int aCol = q* TILE_WIDTH;
    int bRow = q * TILE_WIDTH;
    int bCol = warpN * TILE_WIDTH;

    // Collaborative loading of M and N tiles into shared memory
    if (Row < numARows && (q * TILE_WIDTH + tx) < numAColumns)
      subTileA[ty*TILE_WIDTH + tx] = __float2half(A[Row * numAColumns + q * TILE_WIDTH + tx]);
    else
      subTileA[ty*TILE_WIDTH + tx] = __float2half(0);
    if (Col < numBColumns && (q * TILE_WIDTH + ty) < numBRows)
      subTileB[ty*TILE_WIDTH + tx] = __float2half(in_4d(b, c, input_h + p, input_w + q2));
    else
      subTileB[ty*TILE_WIDTH + tx] = __float2half(0);
    __syncthreads();

    if(aRow < numARows && bRow < numBRows && aCol < numAColumns && bCol < numBColumns){
      wmma::load_matrix_sync(a_frag, subTileA, TILE_WIDTH);
      wmma::load_matrix_sync(b_frag, subTileB, TILE_WIDTH);

      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    __syncthreads();

  }
  __syncthreads();
  // Load in current value of c, scale by beta, and add to result scaled by alpha
  int cRow = warpM * TILE_WIDTH;
  int cCol = warpN * TILE_WIDTH;

  if (cRow < numCRows && cCol < numCColumns)
  {
    for (int i = 0; i < c_frag.num_elements; i++)
    {
      c_frag.x[i] = acc_frag.x[i] + c_frag.x[i];
    }
    wmma::store_matrix_sync(C + cRow * ldc + cCol, c_frag, ldc, wmma::mem_row_major);
  }
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

  // Allocate device_output_ptr with size as a multiple of 16
  const int cCols = (H_out * W_out + TILE_WIDTH - 1) / TILE_WIDTH * TILE_WIDTH;
  const int cRows = (M + TILE_WIDTH - 1) / TILE_WIDTH * TILE_WIDTH;

  cudaMalloc(device_output_ptr, B * cCols * cRows * sizeof(float));
  // cudaMalloc(device_output_ptr, B * M * H_out * W_out * sizeof(float));

  // cudaMalloc(device_output_ptr, B * cCols * cRows * sizeof(float));

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
  int numCRows = (numARows + TILE_WIDTH - 1) / TILE_WIDTH * TILE_WIDTH;
  int numCColumns = (numBColumns + TILE_WIDTH - 1) / TILE_WIDTH * TILE_WIDTH;
  // int numCRows = M;
  // int numCColumns = H_out * W_out;

  int W_size = (numCColumns + TILE_WIDTH - 1) / TILE_WIDTH;
  int H_size = (numCRows + TILE_WIDTH - 1) / TILE_WIDTH;
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 gridDim(W_size, H_size, B);

  matrixMultiplyShared<<<gridDim, blockDim>>>(device_mask, device_input, device_output, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns, H, W, K, S, C);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    std::cout << "CUDA error after kernel launch: " << cudaGetErrorString(error) << std::endl;
    exit(-1);
  }
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
  const int H_out = (H - K) / S + 1;
  const int W_out = (W - K) / S + 1;

  const int numCCols = (H_out * W_out + TILE_WIDTH - 1) / TILE_WIDTH * TILE_WIDTH;

  // Copy the output back to host
  const int hostWidth = H_out * W_out;
  const int hostHeight = B * M;
  const int deviceWidth = numCCols;
  const int deviceHeight = (M + TILE_WIDTH - 1) / TILE_WIDTH * TILE_WIDTH;
  // std::cout << "hostWidth: " << hostWidth << std::endl;
  // std::cout << "hostHeight: " << hostHeight << std::endl;
  // std::cout << "deviceWidth: " << deviceWidth << std::endl;
  // std::cout << "deviceHeight: " << deviceHeight << std::endl;
  

  cudaMemcpy2D(host_output, hostWidth * sizeof(float), device_output, deviceWidth * sizeof(float), hostWidth * sizeof(float), hostHeight, cudaMemcpyDeviceToHost);



  // Free device memory
  cudaFree(device_output);
  cudaFree(device_input);
  cudaFree(device_mask);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    std::cout << "CUDA error in epilog: " << cudaGetErrorString(error) << std::endl;
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

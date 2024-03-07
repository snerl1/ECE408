#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>

#define TILE_WIDTH 16

// kernel to perform shared memory matrix multiplication
__global__ void matrixMultiplyShared(const float *A, const float *input, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns, int H, int W, int K, int S, int C2)
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

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
  /*
  Modify this function to implement the forward pass described in Chapter 16.
  We have added an additional dimension to the tensors to support an entire mini-batch
  The goal here is to be correct AND fast.

  Function paramter definitions:
  output - output
  input - input
  mask - convolution kernel
  B - batch_size (number of images in x)
  M - number of output feature maps
  C - number of input feature maps
  H - input height dimension
  W - input width dimension
  K - kernel height and width (K x K)
  S - stride step length
  */

  const int H_out = (H - K) / S + 1;
  const int W_out = (W - K) / S + 1;

  // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
  // An example use of these macros:
  // float a = in_4d(0,0,0,0)
  // out_4d(0,0,0,0) = a

#define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  // Insert your GPU convolution kernel code here
  int W_size = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;

  int b = blockIdx.z;
  int m = blockIdx.x;
  int h = ((blockIdx.y / W_size) * TILE_WIDTH + threadIdx.y);
  int w = ((blockIdx.y % W_size) * TILE_WIDTH + threadIdx.x);

  int input_h = h * S;
  int input_w = w * S;
  float acc = 0.0f;

  if (input_h < H && input_w < W)
  {
    for (int c = 0; c < C; ++c)
    {
      for (int kh = 0; kh < K; ++kh)
      {
        if (input_h + kh < H)
        {
          for (int kw = 0; kw < K; ++kw)
          {
            if (input_w + kw < W)
              acc += in_4d(b, c, input_h + kh, input_w + kw) * mask_4d(m, c, kh, kw);
          }
        }
      }
    }
  }
  if (h < H_out && w < W_out)
  {
    out_4d(b, m, h, w) = acc;
  }

#undef out_4d
#undef in_4d
#undef mask_4d
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
  int layer = H_out > 40 ? 0 : 1;


  if (layer == 1)
  {
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
  }
  else
  {
    // Set the kernel dimensions and call the kernel
    int W_size = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int H_size = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int Y = H_size * W_size;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(M, Y, B);

    // print blockDim and gridDim
    // std::cout << "blockDim: " << blockDim.x << " " << blockDim.y << " " << blockDim.z << std::endl;

    conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
  }
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

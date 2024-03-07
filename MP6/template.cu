// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                \
  do                                                                 \
  {                                                                  \
    cudaError_t err = stmt;                                          \
    if (err != cudaSuccess)                                          \
    {                                                                \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                    \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err)); \
      return -1;                                                     \
    }                                                                \
  } while (0)

__global__ void postScan(float *blockSums, float *output, int len)
{
  // implement the kernel that adds the accumulative block sums to the appropriate elements of the per-block scan array to complete the scan for all the elements.
  int input_index = HISTOGRAM_LENGTH * blockIdx.x + threadIdx.x;
  if (input_index < len && blockIdx.x > 0)
  {
    output[input_index] = output[input_index] + blockSums[blockIdx.x-1];
  }
}

__global__ void blockScan(float *input, float *output, float *partialSums, int len)
{
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host

  // performs scan on a block to compute the partial block sum
  __shared__ float T[2 * HISTOGRAM_LENGTH];
  int input_index1 = HISTOGRAM_LENGTH * blockIdx.x + threadIdx.x;
  // load data into T
  if (input_index1 < len)
  {
    T[threadIdx.x] = input[input_index1];
  }
  else
  {
    T[threadIdx.x] = 0;
  }

  int stride = 1;
  while (stride < 2 * HISTOGRAM_LENGTH)
  {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index < 2 * HISTOGRAM_LENGTH && (index - stride) >= 0)
      T[index] += T[index - stride];
    stride = stride * 2;
  }

  stride = HISTOGRAM_LENGTH / 2;
  while (stride > 0)
  {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if ((index + stride) < 2 * HISTOGRAM_LENGTH)
      T[index + stride] += T[index];
    stride = stride / 2;
  }

  __syncthreads();

  if (input_index1 < len)
  {
    output[input_index1] = T[threadIdx.x];
  }

  partialSums[blockIdx.x] = T[2 * HISTOGRAM_LENGTH - 1];
}

__global__ void partialSumScan(float *input, float *output, int len)
{
  // performs scan on the partial block sums
  __shared__ float T[2 * HISTOGRAM_LENGTH];
  int input_index1 = HISTOGRAM_LENGTH * blockIdx.x + threadIdx.x;
  // load data into T
  if (input_index1 < len)
  {
    T[threadIdx.x] = input[input_index1];
  }
  else
  {
    T[threadIdx.x] = 0;
  }
  int stride = 1;
  while (stride < 2 * HISTOGRAM_LENGTH)
  {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index < 2 * HISTOGRAM_LENGTH && (index - stride) >= 0)
      T[index] += T[index - stride];
    stride = stride * 2;
  }

  stride = HISTOGRAM_LENGTH / 2;
  while (stride > 0)
  {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if ((index + stride) < 2 * HISTOGRAM_LENGTH)
      T[index + stride] += T[index];
    stride = stride / 2;
  }

  __syncthreads();

  if (input_index1 < len)
  {
    output[input_index1] = T[threadIdx.x];
  }
}

int main(int argc, char **argv)
{
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  int numBlocks = ceil(numElements / (1.0 * HISTOGRAM_LENGTH));

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid1(numBlocks, 1, 1);
  dim3 dimBlock1(HISTOGRAM_LENGTH, 1, 1);

  dim3 dimGrid2(1, 1, 1);
  dim3 dimBlock2(numBlocks, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  float *temp;
  float *partialSums;

  cudaMalloc((void **)&temp, numElements * sizeof(float));
  cudaMalloc((void **)&partialSums, numBlocks * sizeof(float));

  // perform scan on each block
  blockScan<<<dimGrid1, dimBlock1>>>(deviceInput, deviceOutput, temp, numElements);

  // perform scan on the partial block sums
  partialSumScan<<<dimGrid2, dimBlock2>>>(temp, partialSums, numBlocks);

  // add the accumulative block sums to the appropriate elements of the per-block scan array to complete the scan for all the elements.
  postScan<<<dimGrid1, dimBlock1>>>(partialSums, deviceOutput, numElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(temp);
  cudaFree(partialSums);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

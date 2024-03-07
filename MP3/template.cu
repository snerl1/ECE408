
#include <wb.h>

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

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
  const int TILE_WIDTH = 32;
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  // Identify the row and column of the C element to work on
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  float Pvalue = 0;
  // Loop over the A and B tiles required to compute the C element
  for (int q = 0; q < ((numAColumns + TILE_WIDTH - 1) / TILE_WIDTH); ++q)
  {
    // Collaborative loading of M and N tiles into shared memory
    if (Row < numARows && (q * TILE_WIDTH + tx) < numAColumns)
      subTileA[ty][tx] = A[Row * numAColumns + q * TILE_WIDTH + tx];
    else
      subTileB[ty][tx] = 0;
    if (Col < numBColumns && (q * TILE_WIDTH + ty) < numBRows)
      subTileB[ty][tx] = B[(q * TILE_WIDTH + ty) * numBColumns + Col];
    else
      subTileB[ty][tx] = 0;
    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; ++k)
      Pvalue += subTileA[ty][k] * subTileB[k][tx];
    __syncthreads();
  }
  if (Row < numCRows && Col < numCColumns)
    C[Row * numCColumns + Col] = Pvalue;
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCColumns*numCRows*sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc(&deviceA,numAColumns*numARows*sizeof(float));
  cudaMalloc(&deviceB,numBColumns*numBRows*sizeof(float));
  cudaMalloc(&deviceC,numCColumns*numCRows*sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
 cudaMemcpy(deviceA,hostA,numAColumns*numARows*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB,hostB,numBColumns*numBRows*sizeof(float),cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int blockWidth = 32;
  dim3 DimBlock(blockWidth,blockWidth,1);
  dim3 DimGrid((numCColumns + blockWidth - 1) / blockWidth, (numCRows + blockWidth - 1) / blockWidth, 1);

  wbTime_start(Compute, "Performing CUDA computation");

  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows,
                               numAColumns, numBRows,
                               numBColumns,  numCRows,
                                numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here

  cudaMemcpy(hostC,deviceC,numCColumns*numCRows*sizeof(float),cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}

// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 256

//@@ insert code here

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

// cast float to char
__global__ void castToChar(float *input, unsigned char *output, int len)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len)
  {
    output[i] = (unsigned char)(255 * input[i]);
  }
}

// convert RGB to GrayScale
__global__ void convertToGrayScale(unsigned char *input, unsigned char *output, int len)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len)
  {
    output[i] = (unsigned char)(0.21 * input[3 * i] + 0.71 * input[3 * i + 1] + 0.07 * input[3 * i + 2]);
  }
}

// compute the histogram
__global__ void histo_kernel(unsigned char *buffer, long size, unsigned int *histo)
{
  __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];
  // warning: this will not work correctly if there are fewer than 256 threads!
  if (threadIdx.x < HISTOGRAM_LENGTH)
    histo_private[threadIdx.x] = 0;
  __syncthreads();

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  // stride is total number of threads
  int stride = blockDim.x * gridDim.x;
  while (i < size)
  {
    atomicAdd(&(histo_private[buffer[i]]), 1);
    i += stride;
  }
  // wait for all other threads in the block to finish
  __syncthreads();
  if (threadIdx.x < 256)
    atomicAdd(&(histo[threadIdx.x]),
              histo_private[threadIdx.x]);

}

__device__ float p(unsigned int x, unsigned int length)
{
  return x / (1.0 * length);
}

// compute the CDF
// perform scan on the histogram to get the CDF
__global__ void computeCDF(unsigned int *histogram, float *CDF, int len, int size)
{
  // performs scan on the partial block sums
  __shared__ float T[2 * HISTOGRAM_LENGTH];
  int input_index1 = blockDim.x * blockIdx.x + threadIdx.x;
  // load data into T
  if (input_index1 < HISTOGRAM_LENGTH)
  {
    T[threadIdx.x] = p(histogram[input_index1], size);
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
    CDF[input_index1] = T[threadIdx.x];
  }
}

// Define clamp
__device__ float clamp(float color, float min, float max)
{
  float y = color >= min ? color : min;
  return y <= max ? y : max;
}

// Define color correct function
__device__ unsigned char color_correct(unsigned char color, float *cdf, float cdf_min)
{
  return (unsigned char)clamp(255 * (cdf[color] - cdf_min) / (1.0 - cdf_min), 0, 255.0);
}

// Kernel to perform histogram equalization
__global__ void histogram_equalization(unsigned char *input, unsigned char *output, float *cdf, int len)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  float cdf_min_val = cdf[0];
  if (i < len)
  {

    output[i] = color_correct(input[i], cdf, cdf_min_val);
  }
}

// Kernel to cast back to float
__global__ void castToFloat(unsigned char *input, float *output, int len)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len)
  {
    output[i] = (float)(input[i] / 255.0);
  }
}

int main(int argc, char **argv)
{
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;
  float *deviceOutputImageData;

  //@@ Insert more code here
  float *deviceInputImageData;
  unsigned char *deviceCharInputImageData;
  unsigned char *deviceGrayScaleInputImageData;
  unsigned int *deviceHistogram;
  float *deviceCDF;
  unsigned char *deviceCorrectedImageData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  wbCheck(cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceCharInputImageData, imageWidth * imageHeight * imageChannels * sizeof(char)));
  wbCheck(cudaMalloc((void **)&deviceGrayScaleInputImageData, imageWidth * imageHeight * sizeof(char)));
  wbCheck(cudaMalloc((void **)&deviceHistogram, HISTOGRAM_LENGTH * sizeof(int)));
  wbCheck(cudaMalloc((void **)&deviceCDF, HISTOGRAM_LENGTH * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceCorrectedImageData, imageWidth * imageHeight * imageChannels * sizeof(char)));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)));

  wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice));

  // launch kernel to cast float to char
  int numThreads = HISTOGRAM_LENGTH;
  int numElements = imageWidth * imageHeight * imageChannels;
  int numGrayElements = imageWidth * imageHeight;
  int numBlocks =   ceil(numElements / (1.0 * numThreads));
  int numGrayBlocks = ceil(numGrayElements / (1.0 * numThreads));

  castToChar<<<numBlocks, numThreads>>>(deviceInputImageData, deviceCharInputImageData, imageWidth * imageHeight * imageChannels);

  // launch kernel to Convert the image from RGB to GrayScale
  convertToGrayScale<<<numGrayBlocks, numThreads>>>(deviceCharInputImageData, deviceGrayScaleInputImageData, imageWidth * imageHeight);

  // launch kernel to compute the histogram
  histo_kernel<<<numGrayBlocks, numThreads>>>(deviceGrayScaleInputImageData, imageWidth * imageHeight, deviceHistogram);

  // launch kernel to compute the CDF
  computeCDF<<<1, HISTOGRAM_LENGTH>>>(deviceHistogram, deviceCDF, HISTOGRAM_LENGTH, imageWidth * imageHeight);

  // launch kernel to perform histogram equalization
  histogram_equalization<<<numBlocks, numThreads>>>(deviceCharInputImageData, deviceCorrectedImageData, deviceCDF, imageWidth * imageHeight*imageChannels);

  // launch kernel to cast back to float
  castToFloat<<<numBlocks, numThreads>>>(deviceCorrectedImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels);

  // copy the data back to host
  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost));

  wbSolution(args, outputImage);
  wbExport("output_image.ppm", outputImage);

  //@@ insert code here
  cudaFree(deviceInputImageData);
  cudaFree(deviceCharInputImageData);
  cudaFree(deviceGrayScaleInputImageData);
  cudaFree(deviceHistogram);
  cudaFree(deviceCDF);
  cudaFree(deviceCorrectedImageData);
  cudaFree(deviceOutputImageData);

  // get error
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
    return -1;
  }

  return 0;
}

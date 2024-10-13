#include <cstdio>

#include "addition.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

// Device(GPU) pointers
static double *pi_gpu;
static double *pi_cpu;

template <unsigned int blockSize>
__device__ void warpReduce(volatile double* L, int tid){
  if(blockSize>=64) L[tid]+=L[tid+32];
  if(blockSize>=32) L[tid]+=L[tid+16];
  if(blockSize>=16) L[tid]+=L[tid+8];
  if(blockSize>=8 ) L[tid]+=L[tid+4];
  if(blockSize>=4 ) L[tid]+=L[tid+2];
  if(blockSize>=2 ) L[tid]+=L[tid+1];
}

static __device__ double f(double x) { return 4.0 / (1 + x * x); }

__global__ void integral_kernel_v1(double *output, size_t N){
  extern __shared__ double L[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i>=N) return;
  L[tid]=f((double)i/N)/N;
  __syncthreads();

  for(int stride=1; stride<blockDim.x; stride*=2){
    if(tid%(2*stride)==0){
      L[tid]+=L[tid+stride];
    }
    __syncthreads();
  }

  if(tid==0) output[blockIdx.x]=L[0]; 
}

__global__ void integral_kernel_v2(double* output, size_t N){
  
  extern __shared__ double L[];

  int tid=threadIdx.x;
  int i=blockDim.x * blockIdx.x + threadIdx.x;
  double dx=1/(double)N;

  if(i>=N) return;
  L[tid]=f(i*dx)*dx;
  __syncthreads();

  for(int stride=1; stride<blockDim.x; stride*=2){
    int idx = 2*stride*tid;
    if(idx<blockDim.x) L[idx]+=L[idx+stride];
    __syncthreads();  
  }

  if(tid==0) output[blockIdx.x]=L[0];

}

// seqeuntial addressing
__global__ void integral_kernel_v3(double* output, size_t N){

  extern __shared__ double L[];

  int tid=threadIdx.x;
  int idx=blockDim.x * blockIdx.x + threadIdx.x;
  double dx=1/(double)N;

  L[tid]=f(idx*dx)*dx;
  __syncthreads();

  for(int stride=blockDim.x/2; stride>=1; stride>>=1){
    if(tid<stride) L[tid]+=L[tid+stride];
    __syncthreads();
  }

  if(tid==0) output[blockIdx.x]=L[0];

}

__global__ void integral_kernel_v4(double* output, size_t N){
  int tid=threadIdx.x;
  int idx= (blockDim.x*2) * blockIdx.x + threadIdx.x;
  double dx=1/(double)N;
  extern __shared__ double L[];
  
  if(idx+blockDim.x<N){
    L[tid]= ( f(idx*dx)*dx + f((idx+blockDim.x)*dx)*dx );
  }
  __syncthreads();

  for(int stride=blockDim.x/2; stride>0; stride>>=1){
    if(tid<stride) L[tid]+=L[tid+stride];
    __syncthreads();
  }

  if(tid==0) output[blockIdx.x]=L[0];

}

template<unsigned int blockSize>
__global__ void integral_kernel_v5(double* output, size_t N){
  int tid=threadIdx.x;
  int idx= (blockSize*2) * blockIdx.x + threadIdx.x;
  double dx=1/(double)N;
  extern __shared__ double L[];
  
  if(idx+blockSize<N){
    L[tid]=( f(idx*dx)*dx + f((idx+blockSize)*dx)*dx );
  }
  __syncthreads();

  for(int stride=blockSize/2; stride>=32; stride>>=1){
    if(tid<stride) L[tid]+=L[tid+stride];
    __syncthreads();
  }

  if(tid<32) warpReduce<blockSize>(L,tid);

  if(tid==0) output[blockIdx.x]=L[0];
}

template<unsigned int blockSize>
__global__ void integral_kernel_v6(double* output, size_t N){
  int tid=threadIdx.x;
  int idx= (blockSize*2) * blockIdx.x + threadIdx.x;
  double dx=1/(double)N;
  extern __shared__ double L[];
  
  if(idx+blockDim.x<N){
    L[tid]=f(idx*dx)*dx + f((idx+blockSize)*dx)*dx;
  }
  __syncthreads();

  if(blockSize>=512){
    if(tid<256){
      L[tid]+=L[tid+256];
      __syncthreads();  
    }
  }
  if(blockSize>=256){
    if(tid<128){
      L[tid]+=L[tid+64];
      __syncthreads();
    }
  }
  if(blockSize>=128){
    if(tid<64){
      L[tid]+=L[tid+32];
      __syncthreads();
    }
  }

  // warp 단위로 동기적이므로 여기서는 __syncthreads() 호출할 필요 x
  if(tid<32) warpReduce<blockSize>(L,tid);
  if(tid==0) output[blockIdx.x]=L[0];
}

// 한 스레드가 여러개의 elem을 옮겨옴
template<unsigned int blockSize>
__global__ void integral_kernel_v7(double* output, size_t N){
  int tid=threadIdx.x;
  int idx= (blockSize*2) * blockIdx.x + threadIdx.x;
  double dx=1/(double)N;
  int gridSize=blockSize*2*gridDim.x;
  extern __shared__ double L[];
  
  L[tid]=0.0f;
  while(idx+blockSize<N){
    L[tid]+=( f(idx*dx)*dx + f((idx+blockSize)*dx)*dx );
    idx+=gridSize;
  }
  __syncthreads();

  if(blockSize>=512){
    if(tid<256){
      L[tid]+=L[tid+256];
      __syncthreads();  
    }
  }
  if(blockSize>=256){
    if(tid<128){
      L[tid]+=L[tid+64];
      __syncthreads();
    }
  }
  if(blockSize>=128){
    if(tid<64){
      L[tid]+=L[tid+32];
      __syncthreads();
    }
  }

  if(tid<32) warpReduce<blockSize>(L,tid);
  if(tid==0) output[blockIdx.x]=L[0];

}

static double f_naive(double x) { return 4.0 / (1 + x * x); }

double integral_naive(size_t num_intervals) {
  double dx, sum;
  dx = (1.0 / (double) num_intervals);
  sum = 0.0f;
  for (size_t i = 0; i < num_intervals; i++) { sum += f_naive(i * dx) * dx; }
  return sum;
}

const int blocksize=512;

double integral(size_t num_intervals, int v) {
  double pi_value = 0.0;

  dim3 gridDim;
  dim3 blockDim;

  if(v<=3){
    gridDim = dim3(ceil((float)num_intervals/blocksize));
    blockDim = dim3(blocksize);
  }
  else if(v<=6){
    gridDim = dim3(ceil((float)num_intervals/(blocksize)));
    blockDim = dim3(blocksize/2);
  }
  else{
    gridDim = dim3(ceil((float)num_intervals/(blocksize*16)));
    blockDim = dim3(blocksize/2);
  }

  switch(v){
    case 1:
      integral_kernel_v1<<<gridDim, blockDim, sizeof(double) * blocksize>>>(pi_gpu, (int)num_intervals);
      break;
    case 2:
      integral_kernel_v2<<<gridDim, blockDim, sizeof(double) * blocksize>>>(pi_gpu, (int)num_intervals);
      break;
    case 3:
      integral_kernel_v3<<<gridDim, blockDim, sizeof(double) * blocksize>>>(pi_gpu, (int)num_intervals);
      break;
    case 4:
      integral_kernel_v4<<<gridDim, blockDim, sizeof(double) * (blocksize/2)>>>(pi_gpu, (int)num_intervals);
      break;
    case 5:
      integral_kernel_v5<256><<<gridDim, blockDim, sizeof(double) * (blocksize/2)>>>(pi_gpu, (int)num_intervals);
      break;
    case 6:
      integral_kernel_v6<256><<<gridDim, blockDim, sizeof(double) * (blocksize/2)>>>(pi_gpu, (int)num_intervals);
      break;
    case 7:
      integral_kernel_v7<256><<<gridDim, blockDim, sizeof(double) * (blocksize/2)>>>(pi_gpu, (int)num_intervals);
      break;
  }

  cudaMemcpy(pi_cpu, pi_gpu, sizeof(double)*gridDim.x, cudaMemcpyDeviceToHost);

  for(int i=0; i<gridDim.x; i++) pi_value+=pi_cpu[i];

  // NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());

  return pi_value;
  
}

void integral_init(size_t num_intervals, int v) {
  if(v<=3){
    cudaMalloc((void**)&pi_gpu, sizeof(double)*ceil((float)num_intervals/blocksize));
    pi_cpu=(double*)malloc(sizeof(double)*ceil((float)num_intervals/blocksize));
  }
  else if(v<=7){
    cudaMalloc((void**)&pi_gpu, sizeof(double)*ceil((float)num_intervals/(blocksize)));
    pi_cpu=(double*)malloc(sizeof(double)*ceil((float)num_intervals/(blocksize)));
  }

  // NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void integral_cleanup() {
  // Free device memory
  cudaFree(pi_gpu);
  free(pi_cpu);

  // NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

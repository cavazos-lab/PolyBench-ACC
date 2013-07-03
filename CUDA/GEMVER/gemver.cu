/*********************************************************************************/
//
// Polybench kernels implementation on CUDA GPU
//
// Computer & Information Science, University of Delaware
// Author(s):   Sudhee Ayalasomayajula (sudhee1@gmail.com)
//              John Cavazos (cavazos@cis.udel.edu)
//		Scott Grauer Gray(sgrauerg@gmail.com)
//              Robert Searles (rsearles35@aol.com)   
//              Lifan Xu (xulifan@udel.edu)
//
// Contact(s): Lifan Xu (xulifan@udel.edu)
// Reference(s):
//
/*********************************************************************************/

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
#define N 4096

/* Thread block dimensions for kernel 1*/
#define DIM_THREAD_BLOCK_KERNEL_1_X 32
#define DIM_THREAD_BLOCK_KERNEL_1_Y 8

/* Thread block dimensions for kernel 2*/
#define DIM_THREAD_BLOCK_KERNEL_2_X 256
#define DIM_THREAD_BLOCK_KERNEL_2_Y 1

/* Thread block dimensions for kernel 3*/
#define DIM_THREAD_BLOCK_KERNEL_3_X 256
#define DIM_THREAD_BLOCK_KERNEL_3_Y 1

#define ALPHA 43532.0f
#define BETA 12313.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE; 



void gemver(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* z, DATA_TYPE* w, DATA_TYPE* v1,
		DATA_TYPE* v2, DATA_TYPE* u1, DATA_TYPE* u2)
{
	int i,j;
	
  	for (i = 0; i < N; i++)
	{
    	for (j = 0; j < N; j++)
		{
      		A[i*N + j] = A[i*N + j] + u1[i] * v1[j] + u2[i] * v2[j];
		}
	}

  	for (i = 0; i < N; i++)
	{
    	for (j = 0; j < N; j++)
		{
      		x[i] = x[i] + BETA * A[j*N + i] * y[j];
		}
	}

  	for (i = 0; i < N; i++)
	{
    	x[i] = x[i] + z[i];
	}

  	for (i = 0; i < N; i++)
	{
    	for (j = 0; j < N; j++)
		{
      		w[i] = w[i] +  ALPHA * A[i*N + j] * x[j];
		}
	}
}


void init(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* z, DATA_TYPE* w, DATA_TYPE* v1, DATA_TYPE* v2, DATA_TYPE* u1, DATA_TYPE* u2)
{
	int i, j;

  	for (i = 0; i < N; i++)
	{
	    	u1[i] = i;
	    	u2[i] = (i+1)/N/2.0;
	    	v1[i] = (i+1)/N/4.0;
	    	v2[i] = (i+1)/N/6.0;
	    	y[i] = (i+1)/N/8.0;
	    	z[i] = (i+1)/N/9.0;
	    	x[i] = 0.0;
	    	w[i] = 0.0;

    		for (j = 0; j < N; j++)
		{
			A[i*N + j] = ((DATA_TYPE) i*j) / N;
		}
	}
}


void compareResults(DATA_TYPE* w1, DATA_TYPE* w2)
{
	int i, fail;
	fail = 0;
	
	for (i=0; i < N; i++) 
	{
		if (percentDiff(w1[i], w2[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
		{
			fail++;
		}
	}
		
	// Print results
	printf("Number of misses: %d\n", fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}


__global__ void gemver_kernel1(DATA_TYPE *a, DATA_TYPE *v1, DATA_TYPE *v2, DATA_TYPE *u1, DATA_TYPE *u2)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < N) && (j < N))
	{
		a[i * N + j] += u1[i] * v1[j] + u2[i] * v2[j];
	}
}


__global__ void gemver_kernel2(DATA_TYPE *a, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *z)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		int j;
		for(j = 0; j < N; j++) 
		{
			x[i] += BETA * a[j * N + i] * y[j];
		}
		x[i] += z[i];
	}
}


__global__ void gemver_kernel3(DATA_TYPE *a, DATA_TYPE *x, DATA_TYPE *w)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((i >= 0) && (i < N))
	{
		int j;
		for(j = 0; j < N; j++)
		{ 
			w[i] += ALPHA * a[i*N + j] * x[j];
		}
	}
}


void gemverCuda(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* z, DATA_TYPE* w, DATA_TYPE* v1,
				DATA_TYPE* v2, DATA_TYPE* u1, DATA_TYPE* u2, DATA_TYPE* w_outputFromGpu)
{
	double t_start, t_end;

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *x_gpu;
	DATA_TYPE *y_gpu;
	DATA_TYPE *z_gpu;
	DATA_TYPE *v1_gpu;
	DATA_TYPE *v2_gpu;
	DATA_TYPE *u1_gpu;
	DATA_TYPE *u2_gpu;
	DATA_TYPE *w_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * N * N);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * N * N);
	cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&z_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&w_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&v1_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&v2_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&u1_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&u2_gpu, sizeof(DATA_TYPE) * N);
	
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu, y, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(z_gpu, z, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(w_gpu, w, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(v1_gpu, v1, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(v2_gpu, v2, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(u1_gpu, u1, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(u2_gpu, u2, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);

	dim3 block1(DIM_THREAD_BLOCK_KERNEL_1_X, DIM_THREAD_BLOCK_KERNEL_1_Y);
	dim3 grid1((size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_1_X)), (size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_1_Y)));

	dim3 block2(DIM_THREAD_BLOCK_KERNEL_2_X, DIM_THREAD_BLOCK_KERNEL_2_Y);
	dim3 grid2((size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X)), 1);
	
	dim3 block3(DIM_THREAD_BLOCK_KERNEL_3_X, DIM_THREAD_BLOCK_KERNEL_3_Y);
	dim3 grid3((size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_3_X)), 1);
	
	t_start = rtclock();
	gemver_kernel1<<< grid1, block1 >>>(A_gpu,v1_gpu,v2_gpu, u1_gpu, u2_gpu);
	cudaThreadSynchronize();
	gemver_kernel2<<< grid2, block2 >>>(A_gpu,x_gpu,y_gpu, z_gpu);
	cudaThreadSynchronize();
	gemver_kernel3<<< grid3, block3 >>>(A_gpu,x_gpu,w_gpu);
	cudaThreadSynchronize();
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

	cudaMemcpy(w_outputFromGpu, w_gpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost);
	
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(x_gpu);
	cudaFree(y_gpu);
	cudaFree(z_gpu);
	cudaFree(w_gpu);
	cudaFree(v1_gpu);
	cudaFree(v2_gpu);
	cudaFree(u1_gpu);
	cudaFree(u2_gpu);
}
	

int main(int argc, char *argv[])
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;  
	DATA_TYPE* w;  
	DATA_TYPE* w_outputFromGpu;  
	DATA_TYPE* x;  
	DATA_TYPE* y;
	DATA_TYPE* z;
	DATA_TYPE* u1;
	DATA_TYPE* u2;
	DATA_TYPE* v1;
	DATA_TYPE* v2;

	A = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));  
	w = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));  
	w_outputFromGpu = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));  
	x = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));  
	y = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	z = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	u1 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	u2 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	v1 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	v2 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	
	init(A, B, x, y, z, w, v1, v2, u1, u2);
	
	GPU_argv_init();
	gemverCuda(A, B, x, y, z, w, v1, v2, u1, u2, w_outputFromGpu);
	
	t_start = rtclock();
	gemver(A, B, x, y, z, w, v1, v2, u1, u2);
	t_end = rtclock();
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	compareResults(w, w_outputFromGpu);

	free(A);
	free(B);  
	free(w);  
	free(w_outputFromGpu);  
	free(x);  
	free(y);
	free(z);
	free(u1);
	free(u2);
	free(v1);
	free(v2);

 	return 0;
}


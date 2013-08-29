/**
 * gesummv.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#define POLYBENCH_TIME 1

#include "gesummv.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 43532.0f
#define BETA 12313.0f

//#define RUN_ON_CPU


void gesummv(DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_1D(x,N,n), 
		DATA_TYPE POLYBENCH_1D(y,N,n), DATA_TYPE POLYBENCH_1D(tmp,N,n))
{
	int i, j;
	
	for (i = 0; i < N; i++)
	{
		tmp[i] = 0;
		y[i] = 0;
		for (j = 0; j < N; j++)
		{
			tmp[i] = A[i][j] * x[j] + tmp[i];
			y[i] = B[i][j] * x[j] + y[i];
		}
		
		y[i] = ALPHA * tmp[i] + BETA * y[i];
	}
}


void init(DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_1D(x,N,n))
{
  	int i, j;

 	for (i = 0; i < N; i++)
    	{
    		x[i] = ((DATA_TYPE) i) / N;
      	
		for (j = 0; j < N; j++) 
		{
			A[i][j] = ((DATA_TYPE) i*j) / N;
		}
    }
}


void compareResults(DATA_TYPE POLYBENCH_1D(y,N,n), DATA_TYPE POLYBENCH_1D(y_outputFromGpu,N,n))
{
	int i, fail;
	fail = 0;
	
	for (i=0; i<(N); i++) 
	{
		if (percentDiff(y[i], y_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
		{
			fail++;
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}


__global__ void gesummv_kernel(DATA_TYPE* a, DATA_TYPE* b, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		int j;
		for(j = 0; j < N; j++)
		{	
			tmp[i] += a[i * N + j] * x[j];
			y[i] += b[i * N + j] * x[j];
		}
		y[i] = ALPHA * tmp[i] + BETA * y[i];
	}
}

void gesummvCuda(DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_1D(x,N,n), DATA_TYPE POLYBENCH_1D(y,N,n), DATA_TYPE POLYBENCH_1D(tmp,N,n), DATA_TYPE POLYBENCH_1D(y_outputFromGpu,N,n))
{
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *x_gpu;
	DATA_TYPE *y_gpu;
	DATA_TYPE *tmp_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * N * N);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * N * N);
	cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * N);
	
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu, y, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(tmp_gpu, tmp, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((unsigned int)ceil( ((float)N) / ((float)block.x) ), 1);


	/* Start timer. */
  	polybench_start_instruments;

	gesummv_kernel<<< grid, block>>>(A_gpu,B_gpu,x_gpu, y_gpu, tmp_gpu);
	cudaThreadSynchronize();

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	cudaMemcpy(y_outputFromGpu, y_gpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(y,N,n))

{
  int i;

  for (i = 0; i < n; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, y[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
}


int main(int argc, char *argv[])
{
	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,N,N,n,n);
	POLYBENCH_1D_ARRAY_DECL(x,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(y,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(y_outputFromGpu,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(tmp,DATA_TYPE,N,n);

	init(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));
	
	GPU_argv_init();
	gesummvCuda(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(tmp), 
		POLYBENCH_ARRAY(y_outputFromGpu));
	
	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		gesummv(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(tmp));
		
		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;
	
		compareResults(POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(y_outputFromGpu));

	#else //print output to stderr so no dead code elimination

		print_array(N, POLYBENCH_ARRAY(y_outputFromGpu));

	#endif //RUN_ON_CPU


	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);  
	POLYBENCH_FREE_ARRAY(x);  
	POLYBENCH_FREE_ARRAY(y);
	POLYBENCH_FREE_ARRAY(y_outputFromGpu);
	POLYBENCH_FREE_ARRAY(tmp);

	return 0;
}

#include "../../common/polybench.c"

/**
 * syrk.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#define POLYBENCH_TIME 1

#include "syrk.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Declared constant values for alpha and beta (same as values in PolyBench 2.0) */
#define alpha 12435
#define beta 4546

//#define RUN_ON_CPU


void init_arrays(DATA_TYPE POLYBENCH_2D(A, N, M, n, m), DATA_TYPE POLYBENCH_2D(C, N, M, n, m))
{
	int i, j;
	
	for (i = 0; i < N; i++)
    	{
		for (j = 0; j < M; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / N;
		}
		
		for (j = 0; j < N; j++)
		{
			C[i][j] = ((DATA_TYPE) i*j + 2) / N;
		}
	}
}


void syrk(DATA_TYPE POLYBENCH_2D(A, N, M, n, m), DATA_TYPE POLYBENCH_2D(C, N, M, n, m))
{
	int i, j, k;
	
	/*  C := alpha*A*A' + beta*C */
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			C[i][j] *= beta;
		}
	}
	
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			for (k = 0; k < M; k++)
			{
				C[i][j] += alpha * A[i][k] * A[j][k];
			}
		}
	}
}


void compareResults(DATA_TYPE POLYBENCH_2D(C, N, M, n, m), DATA_TYPE POLYBENCH_2D(C_outputFromGpu, N, M, n, m))
{
	int i,j,fail;
	fail = 0;

	// Compare C with D
	for (i=0; i<N; i++)
	{
		for (j=0; j<M; j++)
		{
			if (percentDiff(C[i][j], C_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
	
	return;
}


__global__ void syrk_kernel(DATA_TYPE ALPHA, DATA_TYPE BETA, DATA_TYPE *a, DATA_TYPE *c)
{
	/*  C := alpha*A*A' + beta*C */
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < N) && (j < N))
	{
		c[i * N + j] *= beta;
		int k;		
		for(k=0; k< M; k++)
		{
			c[i * N + j] += alpha * a[i * M + k] * a[j * M + k];
		}
	}
}


void syrkCuda(DATA_TYPE POLYBENCH_2D(A, N, M, n, m), DATA_TYPE POLYBENCH_2D(C, N, M, n, m), DATA_TYPE POLYBENCH_2D(C_outputFromGpu, N, M, n, m))
{
	DATA_TYPE* A_gpu;
	DATA_TYPE* C_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * N * M);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * N * N);
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * N * M, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil(((float)N) / ((float)DIM_THREAD_BLOCK_X))), (size_t)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_Y)));

	/* Start timer. */
  	polybench_start_instruments;

	syrk_kernel<<<grid,block>>>(alpha, beta, A_gpu,C_gpu);
	cudaThreadSynchronize();

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * N * N, cudaMemcpyDeviceToHost);

	cudaFree(A_gpu);
	cudaFree(C_gpu);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n, int m,
		 DATA_TYPE POLYBENCH_2D(C,N,M,n,m))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, C[i][j]);
	if ((i * n + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


int main()
{
	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,M,n,m);
  	POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,N,M,n,m);
  	POLYBENCH_2D_ARRAY_DECL(C_outputFromGpu,DATA_TYPE,N,M,n,m);

	init_arrays(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(C));
	
	GPU_argv_init();	
	syrkCuda(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		syrk(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(C));

		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
  		polybench_stop_instruments;
 		polybench_print_instruments;

		compareResults(POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));

	#else //print output to stderr so no dead code elimination

		print_array(N, M, POLYBENCH_ARRAY(C_outputFromGpu));

	#endif //RUN_ON_CPU


	POLYBENCH_FREE_ARRAY(A);
  	POLYBENCH_FREE_ARRAY(C);
	POLYBENCH_FREE_ARRAY(C_outputFromGpu);

	return 0;
}

#include "../../common/polybench.c"


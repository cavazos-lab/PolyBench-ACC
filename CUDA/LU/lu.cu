/**
 * lu.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#define POLYBENCH_TIME 1

#include "lu.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

//#define RUN_ON_CPU


void lu(DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
	for (int k = 0; k < N; k++)
    {
		for (int j = k + 1; j < N; j++)
		{
			A[k][j] = A[k][j] / A[k][k];
		}

		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
		}
    }
}


void init_array(DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
	int i, j;

	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j + 1) / N;
		}
	}
}


void compareResults(DATA_TYPE POLYBENCH_2D(A_cpu,N,N,n,n), DATA_TYPE POLYBENCH_2D(A_outputFromGpu,N,N,n,n))
{
	int i, j, fail;
	fail = 0;
	
	// Compare a and b
	for (i=0; i<N; i++) 
	{
		for (j=0; j<N; j++) 
		{
			if (percentDiff(A_cpu[i][j], A_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
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


__global__ void lu_kernel1(DATA_TYPE *A, int k)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((j > k) && (j < N))
	{
		A[k*N + j] = A[k*N + j] / A[k*N + k];
	}
}


__global__ void lu_kernel2(DATA_TYPE *A, int k)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i > k) && (j > k) && (i < N) && (j < N))
	{
		A[i*N + j] = A[i*N + j] - A[i*N + k] * A[k*N + j];
	}
}


void luCuda(DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(A_outputFromGpu,N,N,n,n))
{
	DATA_TYPE* AGpu;

	cudaMalloc(&AGpu, N * N * sizeof(DATA_TYPE));
	cudaMemcpy(AGpu, A, N * N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

	dim3 block1(DIM_THREAD_BLOCK_KERNEL_1_X, DIM_THREAD_BLOCK_KERNEL_1_Y);
	dim3 block2(DIM_THREAD_BLOCK_KERNEL_2_X, DIM_THREAD_BLOCK_KERNEL_2_Y);
	dim3 grid1(1, 1, 1);
	dim3 grid2(1, 1, 1);

	/* Start timer. */
  	polybench_start_instruments;

	for (int k = 0; k < N; k++)
	{
		grid1.x = (unsigned int)(ceil((float)(N - (k + 1)) / ((float)block1.x)));
		lu_kernel1<<<grid1, block1>>>(AGpu, k);
		cudaThreadSynchronize();

		grid2.x = (unsigned int)(ceil((float)(N - (k + 1)) / ((float)block2.x)));
		grid2.y = (unsigned int)(ceil((float)(N - (k + 1)) / ((float)block2.y)));
		lu_kernel2<<<grid2, block2>>>(AGpu, k);
		cudaThreadSynchronize();
	}
	
	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	cudaMemcpy(A_outputFromGpu, AGpu, N * N * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
	cudaFree(AGpu);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      fprintf (stderr, DATA_PRINTF_MODIFIER, A[i][j]);
      if ((i * n + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}
	

int main(int argc, char *argv[])
{
	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,N,n,n);
  	POLYBENCH_2D_ARRAY_DECL(A_outputFromGpu,DATA_TYPE,N,N,n,n);

	init_array(POLYBENCH_ARRAY(A));

	GPU_argv_init();
	luCuda(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_outputFromGpu));
	

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		lu(POLYBENCH_ARRAY(A));

		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;
	
		compareResults(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_outputFromGpu));

	#else //print output to stderr so no dead code elimination

		print_array(N, POLYBENCH_ARRAY(A_outputFromGpu));

	#endif //RUN_ON_CPU


	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(A_outputFromGpu);

   	return 0;
}

#include "../../common/polybench.c"


/**
 * jacobi1D.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

#define POLYBENCH_TIME 1

#include "jacobi1D.cuh"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define RUN_ON_CPU


void init_array(int n, DATA_TYPE POLYBENCH_1D(A,N,n), DATA_TYPE POLYBENCH_1D(B,N,n))
{
	int i;

	for (i = 0; i < n; i++)
    	{
		A[i] = ((DATA_TYPE) 4 * i + 10) / N;
		B[i] = ((DATA_TYPE) 7 * i + 11) / N;
    	}
}


void runJacobi1DCpu(int tsteps, int n, DATA_TYPE POLYBENCH_1D(A,N,n), DATA_TYPE POLYBENCH_1D(B,N,n))
{
	for (int t = 0; t < _PB_TSTEPS; t++)
    {
		for (int i = 2; i < _PB_N - 1; i++)
		{
			B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
		}
		
		for (int j = 2; j < _PB_N - 1; j++)
		{
			A[j] = B[j];
		}
    }
}


__global__ void runJacobiCUDA_kernel1(int n, DATA_TYPE* A, DATA_TYPE* B)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((i > 1) && (i < (_PB_N-1)))
	{
		B[i] = 0.33333f * (A[i-1] + A[i] + A[i + 1]);
	}
}


__global__ void runJacobiCUDA_kernel2(int n, DATA_TYPE* A, DATA_TYPE* B)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((j > 1) && (j < (_PB_N-1)))
	{
		A[j] = B[j];
	}
}


void compareResults(int n, DATA_TYPE POLYBENCH_1D(a,N,n), DATA_TYPE POLYBENCH_1D(a_outputFromGpu,N,n), DATA_TYPE POLYBENCH_1D(b,N,n), DATA_TYPE POLYBENCH_1D(b_outputFromGpu,N,n))
{
	int i, fail;
	fail = 0;   

	// Compare a and c
	for (i=0; i < n; i++) 
	{
		if (percentDiff(a[i], a_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
		{
			fail++;
		}
	}

	for (i=0; i < n; i++) 
	{
		if (percentDiff(b[i], b_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
		{	
			fail++;
		}
	}

	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void runJacobi1DCUDA(int tsteps, int n, DATA_TYPE POLYBENCH_1D(A,N,n), DATA_TYPE POLYBENCH_1D(B,N,n), DATA_TYPE POLYBENCH_1D(A_outputFromGpu,N,n), 
			DATA_TYPE POLYBENCH_1D(B_outputFromGpu,N,n))
{
	DATA_TYPE* Agpu;
	DATA_TYPE* Bgpu;

	cudaMalloc(&Agpu, N * sizeof(DATA_TYPE));
	cudaMalloc(&Bgpu, N * sizeof(DATA_TYPE));

	cudaMemcpy(Agpu, A, N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(Bgpu, B, N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
	

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((unsigned int)ceil( ((float)N) / ((float)block.x) ), 1);

	/* Start timer. */
  	polybench_start_instruments;

	for (int t = 0; t < _PB_TSTEPS ; t++)
	{
		runJacobiCUDA_kernel1 <<< grid, block >>> (n, Agpu, Bgpu);
		cudaDeviceSynchronize();
		runJacobiCUDA_kernel2 <<< grid, block>>> (n, Agpu, Bgpu);
		cudaDeviceSynchronize();
	}

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;
	
	cudaMemcpy(A_outputFromGpu, Agpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(B_outputFromGpu, Bgpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost);

	cudaFree(Agpu);
	cudaFree(Bgpu);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(A,N,n))

{
  int i;

  for (i = 0; i < n; i++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i]);
      if (i % 20 == 0) fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}


int main(int argc, char** argv)
{
	/* Retrieve problem size. */
	int n = N;
	int tsteps = TSTEPS;

	POLYBENCH_1D_ARRAY_DECL(a,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(b,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(a_outputFromGpu,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(b_outputFromGpu,DATA_TYPE,N,n);

	init_array(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));

	runJacobi1DCUDA(tsteps, n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(a_outputFromGpu), POLYBENCH_ARRAY(b_outputFromGpu));

	#ifdef RUN_ON_CPU
	
		/* Start timer. */
	  	polybench_start_instruments;
	
		runJacobi1DCpu(tsteps, n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));
	
		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;

		compareResults(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(a_outputFromGpu), POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(b_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(a_outputFromGpu)));

	#endif //RUN_ON_CPU


	POLYBENCH_FREE_ARRAY(a);
	POLYBENCH_FREE_ARRAY(a_outputFromGpu);
	POLYBENCH_FREE_ARRAY(b);
	POLYBENCH_FREE_ARRAY(b_outputFromGpu);

	return 0;
}

#include <polybench.c>
/**
 * jacobi2D.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "jacobi2D.cuh"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define RUN_ON_CPU


void init_array(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
	int i, j;

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			A[i][j] = ((DATA_TYPE) i*(j+2) + 10) / N;
			B[i][j] = ((DATA_TYPE) (i-4)*(j-1) + 11) / N;
		}
	}
}


void runJacobi2DCpu(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
	for (int t = 0; t < _PB_TSTEPS; t++)
	{
    		for (int i = 1; i < _PB_N - 1; i++)
		{
			for (int j = 1; j < _PB_N - 1; j++)
			{
	  			B[i][j] = 0.2f * (A[i][j] + A[i][(j-1)] + A[i][(1+j)] + A[(1+i)][j] + A[(i-1)][j]);
			}
		}
		
    		for (int i = 1; i < _PB_N-1; i++)
		{
			for (int j = 1; j < _PB_N-1; j++)
			{
	  			A[i][j] = B[i][j];
			}
		}
	}
}


__global__ void runJacobiCUDA_kernel1(int n, DATA_TYPE* A, DATA_TYPE* B)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if ((i >= 1) && (i < (_PB_N-1)) && (j >= 1) && (j < (_PB_N-1)))
	{
		B[i*N + j] = 0.2f * (A[i*N + j] + A[i*N + (j-1)] + A[i*N + (1 + j)] + A[(1 + i)*N + j] + A[(i-1)*N + j]);	
	}
}


__global__ void runJacobiCUDA_kernel2(int n, DATA_TYPE* A, DATA_TYPE* B)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((i >= 1) && (i < (_PB_N-1)) && (j >= 1) && (j < (_PB_N-1)))
	{
		A[i*N + j] = B[i*N + j];
	}
}


void compareResults(int n, DATA_TYPE POLYBENCH_2D(a,N,N,n,n), DATA_TYPE POLYBENCH_2D(a_outputFromGpu,N,N,n,n), DATA_TYPE POLYBENCH_2D(b,N,N,n,n), DATA_TYPE POLYBENCH_2D(b_outputFromGpu,N,N,n,n))
{
	int i, j, fail;
	fail = 0;   

	// Compare output from CPU and GPU
	for (i=0; i<n; i++) 
	{
		for (j=0; j<n; j++) 
		{
			if (percentDiff(a[i][j], a_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
        }
	}
  
	for (i=0; i<n; i++) 
	{
       	for (j=0; j<n; j++) 
		{
        		if (percentDiff(b[i][j], b_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
        			fail++;
        		}
       	}
	}

	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void runJacobi2DCUDA(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_2D(A_outputFromGpu,N,N,n,n), DATA_TYPE POLYBENCH_2D(B_outputFromGpu,N,N,n,n))
{
	DATA_TYPE* Agpu;
	DATA_TYPE* Bgpu;

	cudaMalloc(&Agpu, N * N * sizeof(DATA_TYPE));
	cudaMalloc(&Bgpu, N * N * sizeof(DATA_TYPE));
	cudaMemcpy(Agpu, A, N * N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(Bgpu, B, N * N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((unsigned int)ceil( ((float)N) / ((float)block.x) ), (unsigned int)ceil( ((float)N) / ((float)block.y) ));
	
	/* Start timer. */
  	polybench_start_instruments;

	for (int t = 0; t < _PB_TSTEPS; t++)
	{
		runJacobiCUDA_kernel1<<<grid,block>>>(n, Agpu, Bgpu);
		cudaDeviceSynchronize();
		runJacobiCUDA_kernel2<<<grid,block>>>(n, Agpu, Bgpu);
		cudaDeviceSynchronize();
	}

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
  	polybench_print_instruments;
	
	cudaMemcpy(A_outputFromGpu, Agpu, sizeof(DATA_TYPE) * N * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(B_outputFromGpu, Bgpu, sizeof(DATA_TYPE) * N * N, cudaMemcpyDeviceToHost);

	cudaFree(Agpu);
	cudaFree(Bgpu);
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
      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
      if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}


int main(int argc, char** argv)
{
	/* Retrieve problem size. */
	int n = N;
	int tsteps = TSTEPS;

	POLYBENCH_2D_ARRAY_DECL(a,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(b,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(a_outputFromGpu,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(b_outputFromGpu,DATA_TYPE,N,N,n,n);

	init_array(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));
	runJacobi2DCUDA(tsteps, n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(a_outputFromGpu), POLYBENCH_ARRAY(b_outputFromGpu));

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		runJacobi2DCpu(tsteps, n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));
	
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
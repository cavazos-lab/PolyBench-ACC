/**
 * adi.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "adi.cuh"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 2.5

#define GPU_DEVICE 0

#define RUN_ON_CPU


void adi(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_2D(X,N,N,n,n))
{
	for (int t = 0; t < _PB_TSTEPS; t++)
    	{
    		for (int i1 = 0; i1 < _PB_N; i1++)
		{
			for (int i2 = 1; i2 < _PB_N; i2++)
			{
				X[i1][i2] = X[i1][i2] - X[i1][(i2-1)] * A[i1][i2] / B[i1][(i2-1)];
				B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][(i2-1)];
			}
		}

	   	for (int i1 = 0; i1 < _PB_N; i1++)
		{
			X[i1][(N-1)] = X[i1][(N-1)] / B[i1][(N-1)];
		}

	   	for (int i1 = 0; i1 < _PB_N; i1++)
		{
			for (int i2 = 0; i2 < _PB_N-2; i2++)
			{
				X[i1][(N-i2-2)] = (X[i1][(N-2-i2)] - X[i1][(N-2-i2-1)] * A[i1][(N-i2-3)]) / B[i1][(N-3-i2)];
			}
		}

	   	for (int i1 = 1; i1 < _PB_N; i1++)
		{
			for (int i2 = 0; i2 < _PB_N; i2++) 
			{
		  		X[i1][i2] = X[i1][i2] - X[(i1-1)][i2] * A[i1][i2] / B[(i1-1)][i2];
		  		B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[(i1-1)][i2];
			}
		}

	   	for (int i2 = 0; i2 < _PB_N; i2++)
		{
			X[(N-1)][i2] = X[(N-1)][i2] / B[(N-1)][i2];
		}

	   	for (int i1 = 0; i1 < _PB_N-2; i1++)
		{
			for (int i2 = 0; i2 < _PB_N; i2++)
			{
		 	 	X[(N-2-i1)][i2] = (X[(N-2-i1)][i2] - X[(N-i1-3)][i2] * A[(N-3-i1)][i2]) / B[(N-2-i1)][i2];
			}
		}
    }
}


void init_array(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_2D(X,N,N,n,n))
{
  	int i, j;

  	for (i = 0; i < n; i++)
	{
    		for (j = 0; j < n; j++)
      		{
			X[i][j] = ((DATA_TYPE) i*(j+1) + 1) / N;
			A[i][j] = ((DATA_TYPE) (i-1)*(j+4) + 2) / N;
			B[i][j] = ((DATA_TYPE) (i+3)*(j+7) + 3) / N;
      		}
	}
}


void compareResults(int n, DATA_TYPE POLYBENCH_2D(B_cpu,N,N,n,n), DATA_TYPE POLYBENCH_2D(B_fromGpu,N,N,n,n), DATA_TYPE POLYBENCH_2D(X_cpu,N,N,n,n), 
			DATA_TYPE POLYBENCH_2D(X_fromGpu,N,N,n,n))
{
	int i, j, fail;
	fail = 0;
	
	// Compare b and x output on cpu and gpu
	for (i=0; i < n; i++) 
	{
		for (j=0; j < n; j++) 
		{
			if (percentDiff(B_cpu[i][j], B_fromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	for (i=0; i<n; i++) 
	{
		for (j=0; j<n; j++) 
		{
			if (percentDiff(X_cpu[i][j], X_fromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
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


__global__ void adi_kernel1(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X)
{
	int i1 = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((i1 < _PB_N))
	{
		for (int i2 = 1; i2 < _PB_N; i2++)
		{
			X[i1*N + i2] = X[i1*N + i2] - X[i1*N + (i2-1)] * A[i1*N + i2] / B[i1*N + (i2-1)];
			B[i1*N + i2] = B[i1*N + i2] - A[i1*N + i2] * A[i1*N + i2] / B[i1*N + (i2-1)];
		}
	}
}


__global__ void adi_kernel2(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X)
{
	int i1 = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((i1 < _PB_N))
	{
		X[i1*N + (N-1)] = X[i1*N + (N-1)] / B[i1*N + (N-1)];
	}
}
	

__global__ void adi_kernel3(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X)
{
	int i1 = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i1 < _PB_N)
	{
		for (int i2 = 0; i2 < _PB_N-2; i2++)
		{
			X[i1*N + (N-i2-2)] = (X[i1*N + (N-2-i2)] - X[i1*N + (N-2-i2-1)] * A[i1*N + (N-i2-3)]) / B[i1*N + (N-3-i2)];
		}
	}
}


__global__ void adi_kernel4(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X, int i1)
{
	int i2 = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i2 < _PB_N)
	{
		X[i1*N + i2] = X[i1*N + i2] - X[(i1-1)*N + i2] * A[i1*N + i2] / B[(i1-1)*N + i2];
		B[i1*N + i2] = B[i1*N + i2] - A[i1*N + i2] * A[i1*N + i2] / B[(i1-1)*N + i2];
	}
}


__global__ void adi_kernel5(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X)
{
	int i2 = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i2 < _PB_N)
	{
		X[(N-1)*N + i2] = X[(N-1)*N + i2] / B[(N-1)*N + i2];
	}
}


__global__ void adi_kernel6(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X, int i1)
{
	int i2 = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i2 < _PB_N)
	{
		X[(N-2-i1)*N + i2] = (X[(N-2-i1)*N + i2] - X[(N-i1-3)*N + i2] * A[(N-3-i1)*N + i2]) / B[(N-2-i1)*N + i2];
	}
}


void adiCuda(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_2D(X,N,N,n,n), 
	DATA_TYPE POLYBENCH_2D(B_outputFromGpu,N,N,n,n), DATA_TYPE POLYBENCH_2D(X_outputFromGpu,N,N,n,n))
{
	DATA_TYPE* A_gpu;
	DATA_TYPE* B_gpu;
	DATA_TYPE* X_gpu;

	cudaMalloc(&A_gpu, N * N * sizeof(DATA_TYPE));
	cudaMalloc(&B_gpu, N * N * sizeof(DATA_TYPE));
	cudaMalloc(&X_gpu, N * N * sizeof(DATA_TYPE));
	cudaMemcpy(A_gpu, A, N * N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, N * N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(X_gpu, X, N * N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

	dim3 block1(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1);
	dim3 grid1(1, 1, 1);
	grid1.x = (size_t)(ceil( ((float)N) / ((float)block1.x) ));

	/* Start timer. */
  	polybench_start_instruments;

	for (int t = 0; t < _PB_TSTEPS; t++)
	{
		
		adi_kernel1<<<grid1, block1>>>(n, A_gpu, B_gpu, X_gpu);
		cudaThreadSynchronize();
		adi_kernel2<<<grid1, block1>>>(n, A_gpu, B_gpu, X_gpu);
		cudaThreadSynchronize();
		adi_kernel3<<<grid1, block1>>>(n, A_gpu, B_gpu, X_gpu);
		cudaThreadSynchronize();
	
		for (int i1 = 1; i1 < _PB_N; i1++)
		{
			adi_kernel4<<<grid1, block1>>>(n, A_gpu, B_gpu, X_gpu, i1);
			cudaThreadSynchronize();
		}

		adi_kernel5<<<grid1, block1>>>(n, A_gpu, B_gpu, X_gpu);
		cudaThreadSynchronize();
		
		for (int i1 = 0; i1 < _PB_N-2; i1++)
		{
			adi_kernel6<<<grid1, block1>>>(n, A_gpu, B_gpu, X_gpu, i1);
			cudaThreadSynchronize();
		}
	}

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	cudaMemcpy(B_outputFromGpu, B_gpu, N * N * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
	cudaMemcpy(X_outputFromGpu, X_gpu, N * N * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(X_gpu);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(X,N,N,n,n))

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, X[i][j]);
      if ((i * N + j) % 20 == 0) fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}


int main(int argc, char *argv[])
{
	int tsteps = TSTEPS;
	int n = N;

	GPU_argv_init();

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(B_outputFromGpu,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(X,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(X_outputFromGpu,DATA_TYPE,N,N,n,n);

	init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(X));

	adiCuda(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(X), POLYBENCH_ARRAY(B_outputFromGpu), 
		POLYBENCH_ARRAY(X_outputFromGpu));
	

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		adi(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(X));
	
		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;
	
		compareResults(n, POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu), POLYBENCH_ARRAY(X), POLYBENCH_ARRAY(X_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(X_outputFromGpu)));

	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);
	POLYBENCH_FREE_ARRAY(B_outputFromGpu);
	POLYBENCH_FREE_ARRAY(X);
	POLYBENCH_FREE_ARRAY(X_outputFromGpu);

	return 0;
}

#include <polybench.c>
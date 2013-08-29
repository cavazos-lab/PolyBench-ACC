/**
 * gramschmidt.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "gramschmidt.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

//#define RUN_ON_CPU


void gramschmidt(DATA_TYPE POLYBENCH_2D(A,M,N,m,n), DATA_TYPE POLYBENCH_2D(R,M,N,m,n), DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
	int i,j,k;
	DATA_TYPE nrm;
	for (k = 0; k < N; k++)
	{
		nrm = 0;
		for (i = 0; i < M; i++)
		{
			nrm += A[i][k] * A[i][k];
		}
		
		R[k][k] = sqrt(nrm);
		for (i = 0; i < M; i++)
		{
			Q[i][k] = A[i][k] / R[k][k];
		}
		
		for (j = k + 1; j < N; j++)
		{
			R[k][j] = 0;
			for (i = 0; i < M; i++)
			{
				R[k][j] += Q[i][k] * A[i][j];
			}
			for (i = 0; i < M; i++)
			{
				A[i][j] = A[i][j] - Q[i][k] * R[k][j];
			}
		}
	}
}


void init_array(DATA_TYPE POLYBENCH_2D(A,M,N,m,n))
{
	int i, j;

	for (i = 0; i < M; i++)
	{
		for (j = 0; j < N; j++)
		{
			A[i][j] = ((DATA_TYPE) (i+1)*(j+1)) / (M+1);
		}
	}
}


void compareResults(DATA_TYPE POLYBENCH_2D(A,M,N,m,n), DATA_TYPE POLYBENCH_2D(A_outputFromGpu,M,N,m,n))
{
	int i, j, fail;
	fail = 0;

	for (i=0; i < M; i++) 
	{
		for (j=0; j < N; j++) 
		{
			if (percentDiff(A[i][j], A_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
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
	return;
}


__global__ void gramschmidt_kernel1(DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid==0)
	{
		DATA_TYPE nrm = 0.0;
		int i;
		for (i = 0; i < M; i++)
		{
			nrm += a[i * N + k] * a[i * N + k];
		}
      		r[k * N + k] = sqrt(nrm);
	}
}


__global__ void gramschmidt_kernel2(DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < M)
	{	
		q[i * N + k] = a[i * N + k] / r[k * N + k];
	}
}


__global__ void gramschmidt_kernel3(DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if ((j > k) && (j < N))
	{
		r[k*N + j] = 0.0;

		int i;
		for (i = 0; i < M; i++)
		{
			r[k*N + j] += q[i*N + k] * a[i*N + j];
		}
		
		for (i = 0; i < M; i++)
		{
			a[i*N + j] -= q[i*N + k] * r[k*N + j];
		}
	}
}


void gramschmidtCuda(DATA_TYPE POLYBENCH_2D(A,M,N,m,n), DATA_TYPE POLYBENCH_2D(R,M,N,m,n), DATA_TYPE POLYBENCH_2D(Q,M,N,m,n), DATA_TYPE POLYBENCH_2D(A_outputFromGpu,M,N,m,n))
{
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 gridKernel1(1, 1);
	dim3 gridKernel2((size_t)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_X)), 1);
	dim3 gridKernel3((size_t)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_X)), 1);
	
	DATA_TYPE *A_gpu;
	DATA_TYPE *R_gpu;
	DATA_TYPE *Q_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * M * N);
	cudaMalloc((void **)&R_gpu, sizeof(DATA_TYPE) * M * N);
	cudaMalloc((void **)&Q_gpu, sizeof(DATA_TYPE) * M * N);
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * M * N, cudaMemcpyHostToDevice);
	
	/* Start timer. */
  	polybench_start_instruments;
	int k;
	for (k = 0; k < N; k++)
	{
		gramschmidt_kernel1<<<gridKernel1,block>>>(A_gpu, R_gpu, Q_gpu, k);
		cudaThreadSynchronize();
		gramschmidt_kernel2<<<gridKernel2,block>>>(A_gpu, R_gpu, Q_gpu, k);
		cudaThreadSynchronize();
		gramschmidt_kernel3<<<gridKernel3,block>>>(A_gpu, R_gpu, Q_gpu, k);
		cudaThreadSynchronize();
	}
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;
	
	cudaMemcpy(A_outputFromGpu, A_gpu, sizeof(DATA_TYPE) * M * N, cudaMemcpyDeviceToHost);    

	cudaFree(A_gpu);
	cudaFree(R_gpu);
	cudaFree(Q_gpu);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n, DATA_TYPE POLYBENCH_2D(A,M,N,m,n))
{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, A[i][j]);
	if (i % 20 == 0) fprintf (stderr, "\n");
    }

  fprintf (stderr, "\n");
}


int main(int argc, char *argv[])
{
	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,N,m,n);
  	POLYBENCH_2D_ARRAY_DECL(A_outputFromGpu,DATA_TYPE,M,N,m,n);
	POLYBENCH_2D_ARRAY_DECL(R,DATA_TYPE,M,N,m,n);
	POLYBENCH_2D_ARRAY_DECL(Q,DATA_TYPE,M,N,m,n);
	
	init_array(POLYBENCH_ARRAY(A));
	
	GPU_argv_init();

	gramschmidtCuda(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q), POLYBENCH_ARRAY(A_outputFromGpu));

	#ifdef RUN_ON_CPU
	
		/* Start timer. */
	  	polybench_start_instruments;

		gramschmidt(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q));

		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;
	
		compareResults(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_outputFromGpu));
	
	#else //print output to stderr so no dead code elimination

		print_array(M, N, POLYBENCH_ARRAY(A_outputFromGpu));

	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(A_outputFromGpu);
	POLYBENCH_FREE_ARRAY(R);
	POLYBENCH_FREE_ARRAY(Q);  

    	return 0;
}

#include "../../common/polybench.c"



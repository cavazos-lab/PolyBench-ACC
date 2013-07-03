#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size. */
#define TSTEPS 20
#define N 1000

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;


void init_array(DATA_TYPE* A, DATA_TYPE* B)
{
	int i, j;

	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			A[i*N + j] = ((DATA_TYPE) i*(j+2) + 10) / N;
			B[i*N + j] = ((DATA_TYPE) (i-4)*(j-1) + 11) / N;
		}
	}
}


void runJacobi2DCpu(DATA_TYPE* A, DATA_TYPE* B)
{
	for (int t = 0; t < TSTEPS; t++)
	{
    	for (int i = 2; i < N - 1; i++)
		{
			for (int j = 2; j < N - 1; j++)
			{
	  			B[i*N + j] = 0.2f * (A[i*N + j] + A[i*N + (j-1)] + A[i*N + (1+j)] + A[(1+i)*N + j] + A[(i-1)*N + j]);
			}
		}
		
    	for (int i = 2; i < N-1; i++)
		{
			for (int j = 2; j < N-1; j++)
			{
	  			A[i*N + j] = B[i*N + j];
			}
		}
	}
}


__global__ void runJacobiCUDA_kernel1(DATA_TYPE* A, DATA_TYPE* B)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if ((i > 1) && (i < (N-1)) && (j > 1) && (j < (N-1)))
	{
		B[i*N + j] = 0.2f * (A[i*N + j] + A[i*N + (j-1)] + A[i*N + (1 + j)] + A[(1 + i)*N + j] + A[(i-1)*N + j]);	
	}
}


__global__ void runJacobiCUDA_kernel2(DATA_TYPE* A, DATA_TYPE* B)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((i > 1) && (i < (N-1)) && (j > 1) && (j < (N-1)))
	{
		A[i*N + j] = B[i*N + j];
	}
}


void compareResults(DATA_TYPE a[N], DATA_TYPE a_outputFromGpu[N], DATA_TYPE b[N], DATA_TYPE b_outputFromGpu[N])
{
	int i, j, fail;
	fail = 0;   

	// Compare a and c
	for (i=0; i<N; i++) 
	{
		for (j=0; j<N; j++) 
		{
			if (percentDiff(a[i*N + j], a_outputFromGpu[i*N + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
        }
	}
  
	for (i=0; i<N; i++) 
	{
       	for (j=0; j<N; j++) 
		{
        	if (percentDiff(b[i*N + j], b_outputFromGpu[i*N + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
        		fail++;
        	}
       	}
	}

	// Print results
	printf("Number of misses: %d\n", fail);
}


void runJacobi2DCUDA(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* A_outputFromGpu, DATA_TYPE* B_outputFromGpu)
{
	double t_start, t_end;

	DATA_TYPE* Agpu;
	DATA_TYPE* Bgpu;

	cudaMalloc(&Agpu, N * N * sizeof(DATA_TYPE));
	cudaMalloc(&Bgpu, N * N * sizeof(DATA_TYPE));
	cudaMemcpy(Agpu, A, N * N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(Bgpu, B, N * N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((unsigned int)ceil( ((float)N) / ((float)block.x) ), (unsigned int)ceil( ((float)N) / ((float)block.y) ));
	
	t_start = rtclock();

	for (int t = 0; t < TSTEPS; t++)
	{
		runJacobiCUDA_kernel1<<<grid,block>>>(Agpu, Bgpu);
		cudaThreadSynchronize();
		runJacobiCUDA_kernel2<<<grid,block>>>(Agpu, Bgpu);
		cudaThreadSynchronize();
	}

	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

	cudaMemcpy(A_outputFromGpu, Agpu, sizeof(DATA_TYPE) * N * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(B_outputFromGpu, Bgpu, sizeof(DATA_TYPE) * N * N, cudaMemcpyDeviceToHost);

	cudaFree(Agpu);
	cudaFree(Bgpu);
}


int main(int argc, char** argv)
{
	double t_start, t_end;

	DATA_TYPE* a;
	DATA_TYPE* b;
	DATA_TYPE* a_outputFromGpu;
	DATA_TYPE* b_outputFromGpu;

	a = (DATA_TYPE*)malloc(N * N * sizeof(DATA_TYPE));
	b = (DATA_TYPE*)malloc(N * N * sizeof(DATA_TYPE));

	a_outputFromGpu = (DATA_TYPE*)malloc(N * N * sizeof(DATA_TYPE));
	b_outputFromGpu = (DATA_TYPE*)malloc(N * N * sizeof(DATA_TYPE));

	init_array(a, b);
	runJacobi2DCUDA(a, b, a_outputFromGpu, b_outputFromGpu);

	t_start = rtclock();
	runJacobi2DCpu(a, b);
	t_end = rtclock();
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	compareResults(a, a_outputFromGpu, b, b_outputFromGpu);

	free(a);
	free(a_outputFromGpu);
	free(b);
	free(b_outputFromGpu);

	return 0;
}


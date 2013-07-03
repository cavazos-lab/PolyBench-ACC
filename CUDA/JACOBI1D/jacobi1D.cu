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
#define TSTEPS 10000
#define N 4096

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void init_array(DATA_TYPE* A, DATA_TYPE* B)
{
	int i;

	for (i = 0; i < N; i++)
    {
		A[i] = ((DATA_TYPE) 4 * i + 10) / N;
		B[i] = ((DATA_TYPE) 7 * i + 11) / N;
    }
}


void runJacobi1DCpu(DATA_TYPE* A, DATA_TYPE* B)
{
	for (int t = 0; t < TSTEPS; t++)
    {
		for (int i = 2; i < N - 1; i++)
		{
			B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
		}
		
		for (int j = 2; j < N - 1; j++)
		{
			A[j] = B[j];
		}
    }
}


__global__ void runJacobiCUDA_kernel1(DATA_TYPE* A, DATA_TYPE* B)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((i > 1) && (i < (N-1)))
	{
		B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
	}
}


__global__ void runJacobiCUDA_kernel2(DATA_TYPE* A, DATA_TYPE* B)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((j > 1) && (j < (N-1)))
	{
		A[j] = B[j];
	}
}


void compareResults(DATA_TYPE a[N], DATA_TYPE a_outputFromGpu[N], DATA_TYPE b[N], DATA_TYPE b_outputFromGpu[N])
{
	int i, fail;
	fail = 0;   

	// Compare a and c
	for (i=0; i < N; i++) 
	{
		if (percentDiff(a[i], a_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
		{
			fail++;
		}
	}

	for (i=0; i < N; i++) 
	{
		if (percentDiff(b[i], b_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
		{	
			fail++;
		}
	}

	// Print results
	printf("Number of misses: %d\n", fail);
}


void runJacobi1DCUDA(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* A_outputFromGpu, DATA_TYPE* B_outputFromGpu)
{
	double t_start, t_end;

	DATA_TYPE* Agpu;
	DATA_TYPE* Bgpu;

	cudaMalloc(&Agpu, N * sizeof(DATA_TYPE));
	cudaMalloc(&Bgpu, N * sizeof(DATA_TYPE));

	cudaMemcpy(Agpu, A, N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(Bgpu, B, N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
	
	t_start = rtclock();

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((unsigned int)ceil( ((float)N) / ((float)block.x) ), 1);

	
	for (int t = 0; t < TSTEPS ; t++)
	{
		runJacobiCUDA_kernel1 <<< grid, block >>> (Agpu, Bgpu);
		cudaThreadSynchronize();
		runJacobiCUDA_kernel2 <<< grid, block>>> (Agpu, Bgpu);
		cudaThreadSynchronize();
	}

	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

	cudaMemcpy(A_outputFromGpu, Agpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(B_outputFromGpu, Bgpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost);

	cudaFree(Agpu);
	cudaFree(Bgpu);
}


int main(int argc, char** argv)
{
	DATA_TYPE* a;
	DATA_TYPE* b;
	DATA_TYPE* a_outputFromGpu;
	DATA_TYPE* b_outputFromGpu;

	a = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
	b = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));

	a_outputFromGpu = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));
	b_outputFromGpu = (DATA_TYPE*)malloc(N * sizeof(DATA_TYPE));

	init_array(a, b);

	runJacobi1DCUDA(a, b, a_outputFromGpu, b_outputFromGpu);
	
	double t_start, t_end;
	
	t_start = rtclock();
	runJacobi1DCpu(a, b);
	t_end = rtclock();

	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	compareResults(a, a_outputFromGpu, b, b_outputFromGpu);

	free(a);
	free(a_outputFromGpu);
	free(b);
	free(b_outputFromGpu);

	return 0;
}


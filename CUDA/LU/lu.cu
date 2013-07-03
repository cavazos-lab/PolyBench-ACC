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

#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size. */
#define N 2048

/* Thread block dimensions for kernel 1 */
#define DIM_THREAD_BLOCK_KERNEL_1_X 256
#define DIM_THREAD_BLOCK_KERNEL_1_Y 1

/* Thread block dimensions for kernel 2 */
#define DIM_THREAD_BLOCK_KERNEL_2_X 32
#define DIM_THREAD_BLOCK_KERNEL_2_Y 8


/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void lu(DATA_TYPE* A)
{
	for (int k = 0; k < N; k++)
    {
		for (int j = k + 1; j < N; j++)
		{
			A[k*N + j] = A[k*N + j] / A[k*N + k];
		}

		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
			{
				A[i*N + j] = A[i*N + j] - A[i*N + k] * A[k*N + j];
			}
		}
    }
}


void init_array(DATA_TYPE* A)
{
	int i, j;

	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			A[i*N + j] = ((DATA_TYPE) i*j + 1) / N;
		}
	}
}


void compareResults(DATA_TYPE* A_cpu, DATA_TYPE* A_outputFromGpu)
{
	int i, j, fail;
	fail = 0;
	
	// Compare a and b
	for (i=0; i<N; i++) 
	{
		for (j=0; j<(N); j++) 
		{
			if (percentDiff(A_cpu[i*N + j], A_outputFromGpu[i*N + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
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


void luCuda(DATA_TYPE* A, DATA_TYPE* A_outputFromGpu)
{
	double t_start, t_end;

	DATA_TYPE* AGpu;

	cudaMalloc(&AGpu, N * N * sizeof(DATA_TYPE));
	cudaMemcpy(AGpu, A, N * N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

	dim3 block1(DIM_THREAD_BLOCK_KERNEL_1_X, DIM_THREAD_BLOCK_KERNEL_1_Y);
	dim3 block2(DIM_THREAD_BLOCK_KERNEL_2_X, DIM_THREAD_BLOCK_KERNEL_2_Y);
	dim3 grid1(1, 1, 1);
	dim3 grid2(1, 1, 1);

	t_start = rtclock();

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
	
	t_end = rtclock();
	cudaMemcpy(A_outputFromGpu, AGpu, N * N * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	cudaFree(AGpu);
}
	

int main(int argc, char *argv[])
{
	double t_start, t_end;
	
	DATA_TYPE* A;
	DATA_TYPE* A_outputFromGpu;

	A = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));
	A_outputFromGpu = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));

	init_array(A);

	GPU_argv_init();
	luCuda(A, A_outputFromGpu);
	
	t_start = rtclock();
	lu(A);
	t_end = rtclock();
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	compareResults(A, A_outputFromGpu);

	free(A);
	free(A_outputFromGpu);

   	return 0;
}


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
#define PERCENT_DIFF_ERROR_THRESHOLD 2.5

#define GPU_DEVICE 0

/* Problem size. */
#define TSTEPS 1
#define N 1024

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void adi(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X)
{
	for (int t = 0; t < TSTEPS; t++)
    {
    	for (int i1 = 0; i1 < N; i1++)
		{
			for (int i2 = 1; i2 < N; i2++)
			{
				X[i1*N + i2] = X[i1*N + i2] - X[i1*N + (i2-1)] * A[i1*N + i2] / B[i1*N + (i2-1)];
				B[i1*N + i2] = B[i1*N + i2] - A[i1*N + i2] * A[i1*N + i2] / B[i1*N + (i2-1)];
			}
		}

	   	for (int i1 = 0; i1 < N; i1++)
		{
			X[i1*N + (N-1)] = X[i1*N + (N-1)] / B[i1*N + (N-1)];
		}

	   	for (int i1 = 0; i1 < N; i1++)
		{
			for (int i2 = 0; i2 < N-2; i2++)
			{
				X[i1*N + (N-i2-2)] = (X[i1*N + (N-2-i2)] - X[i1*N + (N-2-i2-1)] * A[i1*N + (N-i2-3)]) / B[i1*N + (N-3-i2)];
			}
		}

	   	for (int i1 = 1; i1 < N; i1++)
		{
			for (int i2 = 0; i2 < N; i2++) 
			{
		  		X[i1*N + i2] = X[i1*N + i2] - X[(i1-1)*N + i2] * A[i1*N + i2] / B[(i1-1)*N + i2];
		  		B[i1*N + i2] = B[i1*N + i2] - A[i1*N + i2] * A[i1*N + i2] / B[(i1-1)*N + i2];
			}
		}

	   	for (int i2 = 0; i2 < N; i2++)
		{
			X[(N-1)*N + i2] = X[(N-1)*N + i2] / B[(N-1)*N + i2];
		}

	   	for (int i1 = 0; i1 < N-2; i1++)
		{
			for (int i2 = 0; i2 < N; i2++)
			{
		 	 	X[(N-2-i1)*N + i2] = (X[(N-2-i1)*N + i2] - X[(N-i1-3)*N + i2] * A[(N-3-i1)*N + i2]) / B[(N-2-i1)*N + i2];
			}
		}
    }
}


void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X)
{
  	int i, j;

  	for (i = 0; i < N; i++)
	{
    	for (j = 0; j < N; j++)
      	{
			X[i*N + j] = ((DATA_TYPE) i*(j+1) + 1) / N;
			A[i*N + j] = ((DATA_TYPE) (i-1)*(j+4) + 2) / N;
			B[i*N + j] = ((DATA_TYPE) (i+3)*(j+7) + 3) / N;
      	}
	}
}


void compareResults(DATA_TYPE* B_cpu, DATA_TYPE* B_fromGpu, DATA_TYPE* X_cpu, DATA_TYPE* X_fromGpu)
{
	int i, j, fail;
	fail = 0;
	
	// Compare b and x output on cpu and gpu
	for (i=0; i < N; i++) 
	{
		for (j=0; j < N; j++) 
		{
			if (percentDiff(B_cpu[i*N + j], B_fromGpu[i*N + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
				//printf("1: %f\n 2: %f\n", B_cpu[i*N + j], B_fromGpu[i*N + j]);
			}
		}
	}
	
	for (i=0; i<N; i++) 
	{
		for (j=0; j<(N); j++) 
		{
			if (percentDiff(X_cpu[i*N + j], X_fromGpu[i*N + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
				//printf("1: %f\n 2: %f\n", X_cpu[i*N + j], X_fromGpu[i*N + j]);
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


__global__ void adi_kernel1(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X)
{
	int i1 = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((i1 < N))
	{
		for (int i2 = 1; i2 < N; i2++)
		{
			X[i1*N + i2] = X[i1*N + i2] - X[i1*N + (i2-1)] * A[i1*N + i2] / B[i1*N + (i2-1)];
			B[i1*N + i2] = B[i1*N + i2] - A[i1*N + i2] * A[i1*N + i2] / B[i1*N + (i2-1)];
		}
	}
}


__global__ void adi_kernel2(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X)
{
	int i1 = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((i1 < N))
	{
		X[i1*N + (N-1)] = X[i1*N + (N-1)] / B[i1*N + (N-1)];
	}
}
	

__global__ void adi_kernel3(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X)
{
	int i1 = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((i1 < N))
	{
		for (int i2 = 0; i2 < N-2; i2++)
		{
			X[i1*N + (N-i2-2)] = (X[i1*N + (N-2-i2)] - X[i1*N + (N-2-i2-1)] * A[i1*N + (N-i2-3)]) / B[i1*N + (N-3-i2)];
		}
	}
}


__global__ void adi_kernel4(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X, int i1)
{
	int i2 = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((i2 < N))
	{
		X[i1*N + i2] = X[i1*N + i2] - X[(i1-1)*N + i2] * A[i1*N + i2] / B[(i1-1)*N + i2];
		B[i1*N + i2] = B[i1*N + i2] - A[i1*N + i2] * A[i1*N + i2] / B[(i1-1)*N + i2];
	}
}


__global__ void adi_kernel5(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X)
{
	int i2 = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((i2 < N))
	{
		X[(N-1)*N + i2] = X[(N-1)*N + i2] / B[(N-1)*N + i2];
	}
}


__global__ void adi_kernel6(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X, int i1)
{
	int i2 = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((i2 < N))
	{
		X[(N-2-i1)*N + i2] = (X[(N-2-i1)*N + i2] - X[(N-i1-3)*N + i2] * A[(N-3-i1)*N + i2]) / B[(N-2-i1)*N + i2];
	}
}


void adiCuda(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X, DATA_TYPE* B_outputFromGpu, DATA_TYPE* X_outputFromGpu)
{
	double t_start, t_end;

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

	t_start = rtclock();

	for (int t = 0; t < TSTEPS; t++)
	{
		
		adi_kernel1<<<grid1, block1>>>(A_gpu, B_gpu, X_gpu);
		cudaThreadSynchronize();
		adi_kernel2<<<grid1, block1>>>(A_gpu, B_gpu, X_gpu);
		cudaThreadSynchronize();
		adi_kernel3<<<grid1, block1>>>(A_gpu, B_gpu, X_gpu);
		cudaThreadSynchronize();
	
		for (int i1 = 1; i1 < N; i1++)
		{
			adi_kernel4<<<grid1, block1>>>(A_gpu, B_gpu, X_gpu, i1);
			cudaThreadSynchronize();
		}

		adi_kernel5<<<grid1, block1>>>(A_gpu, B_gpu, X_gpu);
		cudaThreadSynchronize();
		
		for (int i1 = 0; i1 < N-2; i1++)
		{
			adi_kernel6<<<grid1, block1>>>(A_gpu, B_gpu, X_gpu, i1);
			cudaThreadSynchronize();
		}
	}

	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	cudaMemcpy(B_outputFromGpu, B_gpu, N * N * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
	cudaMemcpy(X_outputFromGpu, X_gpu, N * N * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(X_gpu);
}


int main(int argc, char *argv[])
{
	double t_start, t_end;
	
	GPU_argv_init();

	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* B_outputFromGpu;
	DATA_TYPE* X;
	DATA_TYPE* X_outputFromGpu;

	A = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));
	B_outputFromGpu = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));
	X = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));
	X_outputFromGpu = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));

	init_array(A, B, X);

	adiCuda(A, B, X, B_outputFromGpu, X_outputFromGpu);
	
	t_start = rtclock();
	adi(A, B, X);
	t_end = rtclock();
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	compareResults(B, B_outputFromGpu, X, X_outputFromGpu);

	free(A);
	free(B);
	free(B_outputFromGpu);
	free(X);
	free(X_outputFromGpu);

	return 0;
}


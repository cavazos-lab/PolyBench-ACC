/**
 * correlation.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#include <sys/time.h>
#include <cuda.h>

#define POLYBENCH_TIME 1

#include "correlation.cuh"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define GPU_DEVICE 0

#define sqrt_of_array_cell(x,j) sqrt(x[j])

#define FLOAT_N 3214212.01f
#define EPS 0.005f

#define RUN_ON_CPU


void init_arrays(int m, int n, DATA_TYPE POLYBENCH_2D(data, M, N, m, n))
{
	int i, j;
	
	for (i=0; i < m; i++) 
	{
    		for (j=0; j < n; j++) 
		{
       		data[i][j] = ((DATA_TYPE) i*j)/ M;	
       	}
    	}
}


void correlation(int m, int n, DATA_TYPE POLYBENCH_2D(data, M, N, m, n), DATA_TYPE POLYBENCH_1D(mean, M, m), DATA_TYPE POLYBENCH_1D(stddev, M, m),
		DATA_TYPE POLYBENCH_2D(symmat, M, N, m, n))
{
	int i, j, j1, j2;	
	
	// Determine mean of column vectors of input data matrix 
  	for (j = 0; j < _PB_M; j++)
   	{
  		mean[j] = 0.0;

   		for (i = 0; i < _PB_N; i++)
		{
			mean[j] += data[i][j];
   		}
		
		mean[j] /= (DATA_TYPE)FLOAT_N;
   	}

	// Determine standard deviations of column vectors of data matrix. 
  	for (j = 0; j < _PB_M; j++)
   	{
   		stddev[j] = 0.0;
      
		for (i = 0; i < _PB_N; i++)
		{
			stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
		}
		
		stddev[j] /= FLOAT_N;
		stddev[j] = sqrt_of_array_cell(stddev, j);
		stddev[j] = stddev[j] <= EPS ? 1.0 : stddev[j];
	}

 	// Center and reduce the column vectors. 
  	for (i = 0; i < _PB_N; i++)
	{
		for (j = 0; j < _PB_M; j++)
		{
			data[i][j] -= mean[j];
			data[i][j] /= (sqrt(FLOAT_N)*stddev[j]) ;
		}
	}

	// Calculate the m * m correlation matrix. 
  	for (j1 = 0; j1 < _PB_M-1; j1++)
	{	
		symmat[j1][j1] = 1.0;
    
		for (j2 = j1+1; j2 < _PB_M; j2++)
		{
	  		symmat[j1][j2] = 0.0;

	  		for (i = 0; i < _PB_N; i++)
			{
	   			symmat[j1][j2] += (data[i][j1] * data[i][j2]);
			}

	  		symmat[j2][j1] = symmat[j1][j2];
		}
	}
 
	symmat[M-1][M-1] = 1.0;
}


void compareResults(int m, int n, DATA_TYPE POLYBENCH_2D(symmat, M, N, m, n), DATA_TYPE POLYBENCH_2D(symmat_outputFromGpu, M, N, m, n))
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < m; i++)
	{
		for (j=0; j < n; j++)
		{
			if (percentDiff(symmat[i][j], symmat_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
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
}

	
__global__ void mean_kernel(int m, int n, DATA_TYPE *mean, DATA_TYPE *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < _PB_M)
	{
		mean[j] = 0.0;

		int i;
		for(i=0; i < _PB_N; i++)
		{
			mean[j] += data[i*M + j];
		}
		
		mean[j] /= (DATA_TYPE)FLOAT_N;
	}
}


__global__ void std_kernel(int m, int n, DATA_TYPE *mean, DATA_TYPE *std, DATA_TYPE *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < _PB_M)
	{
		std[j] = 0.0;

		int i;
		for(i = 0; i < _PB_N; i++)
		{
			std[j] += (data[i*M + j] - mean[j]) * (data[i*M + j] - mean[j]);
		}
		std[j] /= (FLOAT_N);
		std[j] = sqrt(std[j]);
		if(std[j] <= EPS) 
		{
			std[j] = 1.0;
		}
	}
}


__global__ void reduce_kernel(int m, int n, DATA_TYPE *mean, DATA_TYPE *std, DATA_TYPE *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i < _PB_N) && (j < _PB_M))
	{
		data[i*M + j] -= mean[j];
		data[i*M + j] /= (sqrt(FLOAT_N) * std[j]);
	}
}


__global__ void corr_kernel(int m, int n, DATA_TYPE *symmat, DATA_TYPE *data)
{
	int j1 = blockIdx.x * blockDim.x + threadIdx.x;

	int i, j2;
	if (j1 < (_PB_M-1))
	{
		symmat[j1*M + j1] = 1.0;

		for (j2 = (j1 + 1); j2 < _PB_M; j2++)
		{
			symmat[j1*M + j2] = 0.0;

			for(i = 0; i < _PB_N; i++)
			{
				symmat[j1*M + j2] += data[i*M + j1] * data[i*M + j2];
			}
			symmat[j2*M + j1] = symmat[j1*M + j2];
		}
	}
}


void correlationCuda(int m, int n, DATA_TYPE POLYBENCH_2D(data, M, N, m, n), DATA_TYPE POLYBENCH_1D(mean, M, m), 
			DATA_TYPE POLYBENCH_1D(stddev, M, m), DATA_TYPE POLYBENCH_2D(symmat, M, N, m, n), 
			DATA_TYPE POLYBENCH_2D(symmat_outputFromGpu, M, N, m, n))
{
	DATA_TYPE *data_gpu;
	DATA_TYPE *stddev_gpu;
	DATA_TYPE *mean_gpu;
	DATA_TYPE *symmat_gpu;

	cudaMalloc((void **)&data_gpu, sizeof(DATA_TYPE) * M * N);
	cudaMalloc((void **)&symmat_gpu, sizeof(DATA_TYPE) * M * N);
	cudaMalloc((void **)&stddev_gpu, sizeof(DATA_TYPE) * M);
	cudaMalloc((void **)&mean_gpu, sizeof(DATA_TYPE) * M);
	cudaMemcpy(data_gpu, data, sizeof(DATA_TYPE) * M * N, cudaMemcpyHostToDevice);
	cudaMemcpy(symmat_gpu, symmat, sizeof(DATA_TYPE) * M * N, cudaMemcpyHostToDevice);
	cudaMemcpy(stddev_gpu, stddev, sizeof(DATA_TYPE) * M, cudaMemcpyHostToDevice);
	cudaMemcpy(mean_gpu, mean, sizeof(DATA_TYPE) * M, cudaMemcpyHostToDevice);
		
	dim3 block1(DIM_THREAD_BLOCK_KERNEL_1_X, DIM_THREAD_BLOCK_KERNEL_1_Y);
	dim3 grid1((size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_1_X)), 1);
	
	dim3 block2(DIM_THREAD_BLOCK_KERNEL_2_X, DIM_THREAD_BLOCK_KERNEL_2_Y);
	dim3 grid2((size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X)), 1);
	
	dim3 block3(DIM_THREAD_BLOCK_KERNEL_3_X, DIM_THREAD_BLOCK_KERNEL_3_Y);
	dim3 grid3((size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_3_X)), (size_t)(ceil((float)(N)) / ((float)DIM_THREAD_BLOCK_KERNEL_3_Y)));
	
	dim3 block4(DIM_THREAD_BLOCK_KERNEL_4_X, DIM_THREAD_BLOCK_KERNEL_4_Y);
	dim3 grid4((size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_4_X)), 1);

	/* Start timer. */
  	polybench_start_instruments;

	mean_kernel<<< grid1, block1 >>>(m, n, mean_gpu,data_gpu);
	cudaThreadSynchronize();
	std_kernel<<< grid2, block2 >>>(m, n, mean_gpu,stddev_gpu,data_gpu);
	cudaThreadSynchronize();
	reduce_kernel<<< grid3, block3 >>>(m, n, mean_gpu,stddev_gpu,data_gpu);
	cudaThreadSynchronize();
	corr_kernel<<< grid4, block4 >>>(m, n, symmat_gpu,data_gpu);
	cudaThreadSynchronize();

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	DATA_TYPE valueAtSymmatIndexMTimesMPlus1PlusMPoint = 1.0;
	cudaMemcpy(&(symmat_gpu[(M-1)*M + (M-1)]), &valueAtSymmatIndexMTimesMPlus1PlusMPoint, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

	cudaMemcpy(symmat_outputFromGpu, symmat_gpu, sizeof(DATA_TYPE) * M * N, cudaMemcpyDeviceToHost);
	
	cudaFree(data_gpu);
	cudaFree(symmat_gpu);
	cudaFree(stddev_gpu);
	cudaFree(mean_gpu);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m))

{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      fprintf (stderr, DATA_PRINTF_MODIFIER, symmat[i][j]);
      if ((i * m + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


int main(int argc, char** argv)
{
	int m = M;
	int n = N;

	POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,M,N,m,n);
  	POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
  	POLYBENCH_1D_ARRAY_DECL(stddev,DATA_TYPE,M,m);
	POLYBENCH_2D_ARRAY_DECL(symmat,DATA_TYPE,M,N,m,n);
  	POLYBENCH_2D_ARRAY_DECL(symmat_outputFromGpu,DATA_TYPE,M,N,m,n);
  	
	init_arrays(m, n, POLYBENCH_ARRAY(data));
    
	GPU_argv_init();

	correlationCuda(m, n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(stddev), POLYBENCH_ARRAY(symmat), 
		POLYBENCH_ARRAY(symmat_outputFromGpu));

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		correlation(m, n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(stddev), POLYBENCH_ARRAY(symmat));

		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;
    
		compareResults(m, n, POLYBENCH_ARRAY(symmat), POLYBENCH_ARRAY(symmat_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(symmat_outputFromGpu)));

	#endif //RUN_ON_CPU


	POLYBENCH_FREE_ARRAY(data);
	POLYBENCH_FREE_ARRAY(mean);
	POLYBENCH_FREE_ARRAY(stddev);
	POLYBENCH_FREE_ARRAY(symmat);
	POLYBENCH_FREE_ARRAY(symmat_outputFromGpu);

  	return 0;
}

#include <polybench.c>
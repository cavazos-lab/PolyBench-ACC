/**
 * atax.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>


#include "atax.cuh"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0


#ifndef M_PI
#define M_PI 3.14159
#endif



void init_array(int nx, int ny, DATA_TYPE POLYBENCH_1D(x,NX,nx), DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny))
{
	int i, j;

	for (i = 0; i < nx; i++)
	{
		x[i] = i * M_PI;
		for (j = 0; j < ny; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / NX;
		}
	}
}


void compareResults(int ny, DATA_TYPE POLYBENCH_1D(z,NY,ny), DATA_TYPE POLYBENCH_1D(z_outputFromGpu,NY,ny))
{
	int i, fail;
	fail = 0;

	for (i=0; i<ny; i++)
	{
		if (percentDiff(z[i], z_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
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


__global__ void atax_kernel1(int nx, int ny, DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < _PB_NX)
	{
		tmp[i] = 0;
		int j;
		for(j=0; j < _PB_NY; j++)
		{
			tmp[i] += A[i*NY+j] * x[j];
		}
	}
}

__global__ void atax_kernel2(int nx, int ny, DATA_TYPE *A, DATA_TYPE *y, DATA_TYPE *tmp)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < _PB_NY)
	{
		y[j] = 0;
		int i;
		for(i=0; i < _PB_NX; i++)
		{
			y[j] += A[i*NY+j] * tmp[i];
		}
	}
}


void atax_cpu(int nx, int ny, DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny), DATA_TYPE POLYBENCH_1D(x,NY,ny), DATA_TYPE POLYBENCH_1D(y,NY,ny), 
		DATA_TYPE POLYBENCH_1D(tmp,NX,nx))
{
	int i,j;
	
	for (i= 0; i < _PB_NY; i++)
	{
    		y[i] = 0;
	}
  
	for (i = 0; i < _PB_NX; i++)
 	{
      		tmp[i] = 0;

      		for (j = 0; j < _PB_NY; j++)
		{
			tmp[i] = tmp[i] + A[i][j] * x[j];
		}
		
      		for (j = 0; j < _PB_NY; j++)
		{
			y[j] = y[j] + A[i][j] * tmp[i];
		}
    }
}


void ataxGpu(int nx, int ny, DATA_TYPE POLYBENCH_2D(A, NX, NY,nx,ny), DATA_TYPE POLYBENCH_1D(x,NX,nx), DATA_TYPE POLYBENCH_1D(y,NY,ny), 
		DATA_TYPE POLYBENCH_1D(tmp,NX,nx), DATA_TYPE POLYBENCH_1D(y_outputFromGpu,NY,ny))
{
	DATA_TYPE *A_gpu;
	DATA_TYPE *x_gpu;
	DATA_TYPE *y_gpu;
	DATA_TYPE *tmp_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NX * NY);
	cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * NY);
	cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * NY);
	cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * NX);
	
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice);
	cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu, y, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
	cudaMemcpy(tmp_gpu, tmp, sizeof(DATA_TYPE) * NX, cudaMemcpyHostToDevice);
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NX) / ((float)block.x) )), 1);
	dim3 grid2((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);

	/* Start timer. */
  	polybench_start_instruments;

	atax_kernel1<<< grid1, block >>>(nx, ny, A_gpu,x_gpu,tmp_gpu);
	cudaDeviceSynchronize();
	atax_kernel2<<< grid2, block >>>(nx, ny, A_gpu,y_gpu,tmp_gpu);
	cudaDeviceSynchronize();
	
	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;
	
	cudaMemcpy(y_outputFromGpu, y_gpu, sizeof(DATA_TYPE) * NX, cudaMemcpyDeviceToHost);

	cudaFree(A_gpu);
	cudaFree(x_gpu);
	cudaFree(y_gpu);
	cudaFree(tmp_gpu);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nx, DATA_TYPE POLYBENCH_1D(y,NX,nx))
{
  int i;

  for (i = 0; i < nx; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, y[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
  fprintf (stderr, "\n");
}


int main(int argc, char** argv)
{
	int nx = NX;
	int ny = NY;

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NX,NY,nx,ny);
	POLYBENCH_1D_ARRAY_DECL(x,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(y,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(y_outputFromGpu,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(tmp,DATA_TYPE,NX,nx);

	init_array(nx, ny, POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(A));

	GPU_argv_init();
	ataxGpu(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(tmp), 
		POLYBENCH_ARRAY(y_outputFromGpu));
	
	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		atax_cpu(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(tmp));

		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;

		compareResults(ny, POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(y_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(ny, POLYBENCH_ARRAY(y_outputFromGpu)));

	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(x);
	POLYBENCH_FREE_ARRAY(y);
	POLYBENCH_FREE_ARRAY(y_outputFromGpu);
	POLYBENCH_FREE_ARRAY(tmp);

  	return 0;
}

#include <polybench.c>
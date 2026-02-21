/**
 * doitgen.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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


#include "doitgen.cuh"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0




/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_doitgenCpu(int nr, int nq, int np,
		    DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		    DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
		    DATA_TYPE POLYBENCH_3D(sum,NR,NQ,NP,nr,nq,np))
{
	int r, q, p, s;

	for (r = 0; r < _PB_NR; r++)
	{
		for (q = 0; q < _PB_NQ; q++)  
		{
			for (p = 0; p < _PB_NP; p++)  
			{
				sum[r][q][p] = 0;
				for (s = 0; s < _PB_NP; s++)
					sum[r][q][p] = sum[r][q][p] + A[r][q][s] * C4[s][p];
			}
			for (p = 0; p < _PB_NR; p++)
				A[r][q][p] = sum[r][q][p];
		}
	}

}



/* Array initialization. */
void init_array(int nr, int nq, int np,
		DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np))
{
	int i, j, k;

	for (i = 0; i < nr; i++)
		for (j = 0; j < nq; j++)
			for (k = 0; k < np; k++)
				A[i][j][k] = ((DATA_TYPE) i*j + k) / np;

	for (i = 0; i < np; i++)
		for (j = 0; j < np; j++)
			C4[i][j] = ((DATA_TYPE) i*j) / np;
}


void compareResults(int nr, int nq, int np, DATA_TYPE POLYBENCH_3D(sum,NR,NQ,NP,nr,nq,np), 
			DATA_TYPE POLYBENCH_3D(sum_outputFromGpu,NR,NQ,NP,nr,nq,np))
{
	int fail = 0;
	
	for (int r = 0; r < nr; r++)
	{
    		for (int q = 0; q < nq; q++)  
		{
      			for (int p = 0; p < np; p++)  
			{
				if (percentDiff(sum[r][q][p], sum_outputFromGpu[r][q][p]) > PERCENT_DIFF_ERROR_THRESHOLD)
				{
					fail++;
				}
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


__global__ void doitgen_kernel1(int nr, int nq, int np, DATA_TYPE *sum, DATA_TYPE *A, DATA_TYPE *C4, int r)
{
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	int q = blockIdx.y * blockDim.y + threadIdx.y;

	if ((p < np) && (q < nq))
	{
		sum[r * (nq * np) + q * np + p] = (DATA_TYPE)0.0;
	
		for (int s = 0; s < np; s++)
		{
			sum[r * (nq * np) + q * np + p] = sum[r * (nq * np) + q * np + p] + A[r * (nq * np) + q * np + s] * C4[s * np + p];
		}
	}
}

__global__ void doitgen_kernel2(int nr, int nq, int np, DATA_TYPE *sum, DATA_TYPE *A, DATA_TYPE *C4, int r)
{
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	int q = blockIdx.y * blockDim.y + threadIdx.y;

	if ((p < np) && (q < nq))
	{
		A[r * (nq * np) + q * np + p] = sum[r * (nq * np) + q * np + p];
	}
}

void doitgenCuda(int nr, int nq, int np,
		    DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		    DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
		    DATA_TYPE POLYBENCH_3D(sum_outputFromGpu,NR,NQ,NP,nr,nq,np))
{
	DATA_TYPE* AGpu;
	DATA_TYPE* C4Gpu;
	DATA_TYPE* sumGpu;

	cudaMalloc(&AGpu, nr * nq * np * sizeof(DATA_TYPE));
	cudaMalloc(&C4Gpu, np * np * sizeof(DATA_TYPE));
	cudaMalloc(&sumGpu, nr * nq * np * sizeof(DATA_TYPE));

	cudaMemcpy(AGpu, A, nr * nq * np * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(C4Gpu, C4, np * np * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(sumGpu, sum_outputFromGpu, nr * nq * np * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((unsigned int)ceil( ((float)np) / ((float)block.x) ), (unsigned int)ceil( ((float)nr) / ((float)block.y) ));

	/* Start timer. */
	polybench_start_instruments;	

	for (int r = 0; r < nr; r++)
	{
		doitgen_kernel1 <<<grid, block>>> (nr, nq, np, sumGpu, AGpu, C4Gpu, r);
		cudaDeviceSynchronize();
		doitgen_kernel2 <<<grid, block>>> (nr, nq, np, sumGpu, AGpu, C4Gpu, r);
		cudaDeviceSynchronize();
	}

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
	polybench_print_instruments;
		
	cudaMemcpy(sum_outputFromGpu, sumGpu, NR * NQ * NP * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

	cudaFree(AGpu);
	cudaFree(C4Gpu);
	cudaFree(sumGpu);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nr, int nq, int np,
		 DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np))
{
	int i, j, k;

	for (i = 0; i < nr; i++)
	{
		for (j = 0; j < nq; j++)
		{
			for (k = 0; k < np; k++) 
			{
				fprintf (stderr, DATA_PRINTF_MODIFIER, A[i][j][k]);
				if (i % 20 == 0) fprintf (stderr, "\n");
			}
		}
	}
	fprintf (stderr, "\n");
}
	

int main(int argc, char *argv[])
{
	/* Retrieve problem size. */
	int nr = NR;
	int nq = NQ;
	int np = NP;

	/* Variable declaration/allocation. */
	POLYBENCH_3D_ARRAY_DECL(A,DATA_TYPE,NR,NQ,NP,nr,nq,np);
	POLYBENCH_3D_ARRAY_DECL(sum,DATA_TYPE,NR,NQ,NP,nr,nq,np);
	POLYBENCH_3D_ARRAY_DECL(sum_outputFromGpu,DATA_TYPE,NR,NQ,NP,nr,nq,np);
	POLYBENCH_2D_ARRAY_DECL(C4,DATA_TYPE,NP,NP,np,np);

	/* Initialize array(s). */
	init_array (nr, nq, np,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(C4));

	doitgenCuda(nr, nq, np,
		POLYBENCH_ARRAY(A),
		POLYBENCH_ARRAY(C4),
		POLYBENCH_ARRAY(sum_outputFromGpu));


	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		/* Run kernel on CPU */
		kernel_doitgenCpu(nr, nq, np,
		  POLYBENCH_ARRAY(A),
		  POLYBENCH_ARRAY(C4),
		  POLYBENCH_ARRAY(sum));	
		
		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
  		polybench_stop_instruments;
	 	polybench_print_instruments;
	
		compareResults(nr, nq, np, POLYBENCH_ARRAY(sum), POLYBENCH_ARRAY(sum_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(nr, nq, np, POLYBENCH_ARRAY(sum_outputFromGpu)));

	#endif //RUN_ON_CPU

	/* Garbage collection */
	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(sum);
	POLYBENCH_FREE_ARRAY(sum_outputFromGpu);
	POLYBENCH_FREE_ARRAY(C4);	
    
	return 0;
}

#include <polybench.c>
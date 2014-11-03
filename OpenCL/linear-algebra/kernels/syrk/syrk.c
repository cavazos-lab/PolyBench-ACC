/**
 * syrk.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define POLYBENCH_TIME 1

//select the OpenCL device to use (can be GPU, CPU, or Accelerator such as Intel Xeon Phi)
#define OPENCL_DEVICE_SELECTION CL_DEVICE_TYPE_GPU

#include "syrk.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define MAX_SOURCE_SIZE (0x100000)

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

char str_temp[1024];

DATA_TYPE acc;

cl_platform_id platform_id;
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel1;
cl_kernel clKernel2;
cl_kernel clKernel3;
cl_command_queue clCommandQue;
cl_program clProgram;

cl_mem a_mem_obj;
cl_mem c_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;

#ifndef RUN_ON_CPU
#define RUN_ON_CPU 1
#endif

#define KERNEL_FILE "syrk.cl"

#include "ocl_common.c"


void compareResults(int ni, DATA_TYPE POLYBENCH_2D(C, NI, NI, ni, ni), DATA_TYPE POLYBENCH_2D(C_outputFromGpu, NI, NI, ni, ni))
{
	int i,j,fail;
	fail = 0;

	// Compare C with D
	for (i=0; i<ni; i++)
	{
		for (j=0; j<ni; j++)
		{
			if (percentDiff(C[i][j], C_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void init_arrays(int ni, int nj,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,NI,NI,ni,ni),
		DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj))
{
	int i, j;

	*alpha = 32412;
	*beta = 2123;
	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < nj; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / ni;
		}
	}

	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < ni; j++)
		{
			C[i][j] = ((DATA_TYPE) i*j) / ni;
		}
	}
}

void cl_mem_init(DATA_TYPE POLYBENCH_2D(A,NJ,NI,nj,ni), DATA_TYPE POLYBENCH_2D(C,NJ,NI,nj,ni))
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NI * NJ, NULL, &errcode);
	c_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NI * NJ, NULL, &errcode);
	
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NJ, A, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, c_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NJ, C, 0, NULL, NULL);
	if(errcode != CL_SUCCESS)printf("Error in writing buffers\n");
}

void cl_launch_kernel(int ni, int nj, DATA_TYPE alpha, DATA_TYPE beta)
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)NJ) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)NI) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&c_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 2, sizeof(DATA_TYPE), (void *)&alpha);
	errcode |= clSetKernelArg(clKernel1, 3, sizeof(DATA_TYPE), (void *)&beta);
	errcode |= clSetKernelArg(clKernel1, 4, sizeof(int), (void *)&ni);
	errcode |= clSetKernelArg(clKernel1, 5, sizeof(int), (void *)&nj);

	if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel1\n");
	clFinish(clCommandQue);
}


void cl_clean_up()
{
	// Clean up
	errcode = clFlush(clCommandQue);
	errcode = clFinish(clCommandQue);
	errcode = clReleaseKernel(clKernel1);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(c_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


void syrk(int ni, int nj, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj), DATA_TYPE POLYBENCH_2D(C, NI, NI, ni, ni))
{
	int i, j, k;
	
	/*  C := alpha*A*A' + beta*C */
	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NI; j++)
		{
			C[i][j] *= beta;
		}
	}
	
	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NI; j++)
		{
			for (k = 0; k < _PB_NJ; k++)
			{
				C[i][j] += alpha * A[i][k] * A[j][k];
			}
		}
	}
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, DATA_TYPE POLYBENCH_2D(C,NI,NI,ni,ni))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < ni; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, C[i][j]);
	if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


int main(int argc, char *argv[])
{
	/* Retrieve problem size. */
	int ni = NI;
	int nj = NJ;

	/* Variable declaration/allocation. */
	DATA_TYPE alpha;
	DATA_TYPE beta;

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NJ,ni,nj);
  	POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NI,ni,ni);
  	POLYBENCH_2D_ARRAY_DECL(C_outputFromGpu,DATA_TYPE,NI,NI,ni,ni);

	init_arrays(ni, nj, &alpha, &beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));
	read_cl_file();
	cl_initialization();
	cl_load_prog();
		
	// Create the OpenCL kernel
	clKernel1 = clCreateKernel(clProgram, "syrk_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");
	clFinish(clCommandQue);

	/* Start timer. */
  	polybench_start_instruments;

	cl_mem_init(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(C));
	cl_launch_kernel(ni, nj, alpha, beta);

	errcode = clEnqueueReadBuffer(clCommandQue, c_mem_obj, CL_TRUE, 0, NI * NJ * sizeof(DATA_TYPE), POLYBENCH_ARRAY(C_outputFromGpu), 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");  

	/* Stop and print timer. */
#if VERBOSE == 1
	printf("GPU Time in seconds:\n");
#endif
  	polybench_stop_instruments;
 	polybench_print_instruments;

	#if RUN_ON_CPU == 1

		/* Start timer. */
	  	polybench_start_instruments;

		syrk(ni, nj, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(C));

		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;

		compareResults(ni, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(ni, POLYBENCH_ARRAY(C_outputFromGpu)));

	#endif //RUN_ON_CPU

	cl_clean_up();
	
	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(C);
	POLYBENCH_FREE_ARRAY(C_outputFromGpu);

	return 0;
}

#include <polybench.c>

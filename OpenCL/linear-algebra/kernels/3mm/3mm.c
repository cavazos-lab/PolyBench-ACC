/**
 * 3mm.c: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "3mm.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.001

#define MAX_SOURCE_SIZE (0x100000)

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif


char str_temp[1024];

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
cl_mem b_mem_obj;
cl_mem c_mem_obj;
cl_mem d_mem_obj;
cl_mem e_mem_obj;
cl_mem f_mem_obj;
cl_mem g_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;

#ifndef RUN_ON_CPU
#define RUN_ON_CPU 1
#endif

#define KERNEL_FILE "3mm.cl"

#include "ocl_common.c"


void compareResults(int ni, int nl, DATA_TYPE POLYBENCH_2D(G, NI, NL, ni, nl), DATA_TYPE POLYBENCH_2D(G_outputFromGpu, NI, NL, ni, nl))
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < ni; i++)
	{
		for (j=0; j < nl; j++)
		{
			if (percentDiff(G[i][j], G_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;				
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void init_array(int ni, int nj, int nk, int nl, int nm, DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk), DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj), 
		DATA_TYPE POLYBENCH_2D(C, NJ, NM, nj, nm), DATA_TYPE POLYBENCH_2D(D, NM, NL, nm, nl))
{
	int i, j;

	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < nk; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / ni;
		}
	}
  
	for (i = 0; i < nk; i++)
	{
		for (j = 0; j < nj; j++)
		{
			B[i][j] = ((DATA_TYPE) i*(j+1)) / nj;
		}
	}
  
	for (i = 0; i < nj; i++)
	{
		for (j = 0; j < nm; j++)
		{
			C[i][j] = ((DATA_TYPE) i*(j+3)) / nl;
		}
	}
  
	for (i = 0; i < nm; i++)
	{
		for (j = 0; j < nl; j++)
		{
			D[i][j] = ((DATA_TYPE) i*(j+2)) / nk;
		}
	}
}

void cl_mem_init(
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
		DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY, sizeof(DATA_TYPE) * NI * NK, NULL, &errcode);
	b_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY, sizeof(DATA_TYPE) * NK * NJ, NULL, &errcode);
	c_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NJ * NM, NULL, &errcode);
	d_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NM * NL, NULL, &errcode);
	e_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NI * NJ, NULL, &errcode);
	f_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NJ * NL, NULL, &errcode);
	g_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NI * NL, NULL, &errcode);
		
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NK, A, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NK * NJ, B, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, c_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NJ * NM, C, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, d_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NM * NL, D, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, e_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NJ, E, 0, NULL, NULL);	
	errcode = clEnqueueWriteBuffer(clCommandQue, f_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NJ * NL, F, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, g_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NL, G, 0, NULL, NULL);
	if(errcode != CL_SUCCESS)printf("Error in writing buffers\n");
 }

void cl_launch_kernel(int ni, int nj, int nk, int nl, int nm)
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)NJ) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)NI) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&e_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 3, sizeof(int), (void *)&ni);
	errcode |= clSetKernelArg(clKernel1, 4, sizeof(int), (void *)&nj);
	errcode |= clSetKernelArg(clKernel1, 5, sizeof(int), (void *)&nk);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");
	// Execute the OpenCL kernel

	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);	
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clEnqueueBarrier(clCommandQue);

	globalWorkSize[0] = (size_t)ceil(((float)NL) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)NJ) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;

	errcode =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&c_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&d_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void *)&f_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 3, sizeof(int), (void *)&nj);
	errcode |= clSetKernelArg(clKernel2, 4, sizeof(int), (void *)&nl);
	errcode |= clSetKernelArg(clKernel2, 5, sizeof(int), (void *)&nm);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");
	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);	
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clEnqueueBarrier(clCommandQue);

	globalWorkSize[0] = (size_t)ceil(((float)NL) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)NI) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;

	errcode =  clSetKernelArg(clKernel3, 0, sizeof(cl_mem), (void *)&e_mem_obj);
	errcode |= clSetKernelArg(clKernel3, 1, sizeof(cl_mem), (void *)&f_mem_obj);
	errcode |= clSetKernelArg(clKernel3, 2, sizeof(cl_mem), (void *)&g_mem_obj);
	errcode |= clSetKernelArg(clKernel3, 3, sizeof(int), (void *)&ni);
	errcode |= clSetKernelArg(clKernel3, 4, sizeof(int), (void *)&nl);
	errcode |= clSetKernelArg(clKernel3, 5, sizeof(int), (void *)&nj);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");
	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel3, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);	
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clFinish(clCommandQue);
}

void cl_clean_up()
{
	// Clean up
	errcode = clFlush(clCommandQue);
	errcode = clFinish(clCommandQue);
	errcode = clReleaseKernel(clKernel1);
	errcode = clReleaseKernel(clKernel2);
	errcode = clReleaseKernel(clKernel3);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(b_mem_obj);
	errcode = clReleaseMemObject(c_mem_obj);
	errcode = clReleaseMemObject(d_mem_obj);
	errcode = clReleaseMemObject(e_mem_obj);
	errcode = clReleaseMemObject(f_mem_obj);
	errcode = clReleaseMemObject(g_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


/* Main computational kernel on CPU */
void mm3_cpu(int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
		DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
	int i, j, k;

	/* E := A*B */
	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NJ; j++)
		{
			E[i][j] = 0;
			for (k = 0; k < _PB_NK; ++k)
			{
				E[i][j] += A[i][k] * B[k][j];
			}
		}
	}

	/* F := C*D */
	for (i = 0; i < _PB_NJ; i++)
	{
		for (j = 0; j < _PB_NL; j++)
		{
			F[i][j] = 0;
			for (k = 0; k < _PB_NM; ++k)
			{
				F[i][j] += C[i][k] * D[k][j];
			}
		}
	}

	/* G := E*F */
	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NL; j++)
		{
			G[i][j] = 0;
			for (k = 0; k < _PB_NJ; ++k)
			{
				G[i][j] += E[i][k] * F[k][j];
			}
		}
	}
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, G[i][j]);
	if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


int main(int argc, char *argv[])
{
	int ni = NI;
	int nj = NJ;
	int nk = NK;
	int nl = NL;
	int nm = NM;

	/* Variable declaration/allocation. */
	POLYBENCH_2D_ARRAY_DECL(E, DATA_TYPE, NI, NJ, ni, nj);
	POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
	POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
	POLYBENCH_2D_ARRAY_DECL(F, DATA_TYPE, NJ, NL, nj, nl);
	POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NM, nj, nm);
	POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NM, NL, nm, nl);
	POLYBENCH_2D_ARRAY_DECL(G, DATA_TYPE, NI, NL, ni, nl);
	POLYBENCH_2D_ARRAY_DECL(G_outputFromGpu, DATA_TYPE, NI, NL, ni, nl);

	init_array(ni, nj, nk, nl, nm, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));
  int i, j;

  for (i = 0; i < 10; i++) {
    for (j = 0; j < 10; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, A[i][j]);
    }
fprintf (stderr, "\n");
  }
  fprintf (stderr, "\n");
  for (i = 0; i < 10; i++) {
    for (j = 0; j < 10; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, B[i][j]);
    }
fprintf (stderr, "\n");
  }
  fprintf (stderr, "\n");
  for (i = 0; i < 10; i++) {
    for (j = 0; j < 10; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, C[i][j]);
    }
fprintf (stderr, "\n");
  }
  fprintf (stderr, "\n");
  for (i = 0; i < 10; i++) {
    for (j = 0; j < 10; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, D[i][j]);
    }
fprintf (stderr, "\n");
  }
  fprintf (stderr, "\n");


	read_cl_file();
	cl_initialization();
	cl_load_prog();
		
	// Create the OpenCL kernels
	clKernel1 = clCreateKernel(clProgram, "mm3_kernel1", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
	clKernel2 = clCreateKernel(clProgram, "mm3_kernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
	clKernel3 = clCreateKernel(clProgram, "mm3_kernel3", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
	clFinish(clCommandQue);

	/* Start timer. */
  	polybench_start_instruments;

	cl_mem_init(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(E), POLYBENCH_ARRAY(F), POLYBENCH_ARRAY(G_outputFromGpu));

	cl_launch_kernel(ni, nj, nk, nl, nm);

	errcode = clEnqueueReadBuffer(clCommandQue, g_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NL, POLYBENCH_ARRAY(G_outputFromGpu), 0, NULL, NULL);
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

		mm3_cpu(ni, nj, nk, nl, nm, POLYBENCH_ARRAY(E), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(F), POLYBENCH_ARRAY(C), 
			POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(G));

		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;

		compareResults(ni, nl, POLYBENCH_ARRAY(G), POLYBENCH_ARRAY(G_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(ni, nl, POLYBENCH_ARRAY(G_outputFromGpu)));

	#endif //RUN_ON_CPU

  for (i = 0; i < 10; i++) {
    for (j = 0; j < 10; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, G[i][j]);
    }
fprintf (stderr, "\n");
  }
  fprintf (stderr, "\n");

  for (i = 0; i < 10; i++) {
    for (j = 0; j < 10; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, G_outputFromGpu[i][j]);
    }fprintf (stderr, "\n");
  }
  fprintf (stderr, "\n");

	cl_clean_up();

	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);
	POLYBENCH_FREE_ARRAY(C);
	POLYBENCH_FREE_ARRAY(D);
	POLYBENCH_FREE_ARRAY(E);
	POLYBENCH_FREE_ARRAY(F);
	POLYBENCH_FREE_ARRAY(G);
	POLYBENCH_FREE_ARRAY(G_outputFromGpu);

	return 0;
}

#include <polybench.c>

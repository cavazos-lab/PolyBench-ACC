/**
 * bicg.c: This file is part of the PolyBench/GPU 1.0 test suite.
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


#include "bicg.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define MAX_SOURCE_SIZE (0x100000)

#ifndef M_PI
#define M_PI 3.14159
#endif

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
cl_command_queue clCommandQue;
cl_program clProgram;

cl_mem a_mem_obj;
cl_mem r_mem_obj;
cl_mem p_mem_obj;
cl_mem q_mem_obj;
cl_mem s_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;



void compareResults(int nx, int ny, DATA_TYPE POLYBENCH_1D(s,NY,ny), DATA_TYPE POLYBENCH_1D(s_outputFromGpu,NY,ny), 
		DATA_TYPE POLYBENCH_1D(q,NX,nx), DATA_TYPE POLYBENCH_1D(q_outputFromGpu,NX,nx))
{
	int i,fail;
	fail = 0;

	// Compare s with s_cuda
	for (i=0; i<nx; i++)
	{
		if (percentDiff(q[i], q_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}
	}

	for (i=0; i<ny; i++)
	{
		if (percentDiff(s[i], s_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}		
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen("bicg.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init_array(int nx, int ny, DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny), DATA_TYPE POLYBENCH_1D(p,NY,ny), DATA_TYPE POLYBENCH_1D(r,NX,nx))
{
	int i, j;
	
	for (i = 0; i < ny; i++)
	{
    		p[i] = i * M_PI;
	}

	for (i = 0; i < nx; i++)
	{
    		r[i] = i * M_PI;

    		for (j = 0; j < ny; j++)
		{
      			A[i][j] = ((DATA_TYPE) i*j) / NX;
		}
 	}
}


void cl_initialization()
{	
	// Get platform and device information
	errcode = clGetPlatformIDs(1, &platform_id, &num_platforms);
	if(errcode == CL_SUCCESS) printf("number of platforms is %d\n",num_platforms);
	else printf("Error getting platform IDs\n");

	errcode = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform name is %s\n",str_temp);
	else printf("Error getting platform name\n");

	errcode = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform version is %s\n",str_temp);
	else printf("Error getting platform version\n");

	errcode = clGetDeviceIDs( platform_id, OPENCL_DEVICE_SELECTION, 1, &device_id, &num_devices);
	if(errcode == CL_SUCCESS) printf("number of devices is %d\n", num_devices);
	else printf("Error getting device IDs\n");

	errcode = clGetDeviceInfo(device_id,CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("device name is %s\n",str_temp);
	else printf("Error getting device name\n");
	
	// Create an OpenCL context
	clGPUContext = clCreateContext( NULL, 1, &device_id, NULL, NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating context\n");
 
	//Create a command-queue
	clCommandQue = clCreateCommandQueue(clGPUContext, device_id, 0, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating command queue\n");
}


void cl_mem_init(DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny), DATA_TYPE POLYBENCH_1D(r,NX,nx), DATA_TYPE POLYBENCH_1D(s,NX,nx), 
	DATA_TYPE POLYBENCH_1D(p,NX,nx), DATA_TYPE POLYBENCH_1D(q,NX,nx))
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NX * NY, NULL, &errcode);
	r_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NX, NULL, &errcode);
	s_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NX, NULL, &errcode);
	p_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NX, NULL, &errcode);
	q_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NX, NULL, &errcode);
		
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");
	
	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX * NY, A, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, r_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX, r, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, s_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX, s, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, p_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX, p, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, q_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX, q, 0, NULL, NULL);
	if(errcode != CL_SUCCESS)printf("Error in writing buffers\n");
 }

void cl_load_prog()
{
	// Create a program from the kernel source
	clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);

	if(errcode != CL_SUCCESS) printf("Error in creating program\n");

	// Build the program
	errcode = clBuildProgram(clProgram, 1, &device_id, NULL, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in building program\n");
		
	// Create the 1st OpenCL kernel
	clKernel1 = clCreateKernel(clProgram, "bicgKernel1", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");

	// Create the 2nd OpenCL kernel
	clKernel2 = clCreateKernel(clProgram, "bicgKernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");

	clFinish(clCommandQue);
}

void cl_launch_kernel(int nx, int ny)
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)NX) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = 1;

	/* Start timer. */
  	polybench_start_instruments;
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&p_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&q_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 3, sizeof(int), &nx);
        errcode |= clSetKernelArg(clKernel1, 4, sizeof(int), &ny);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

	// Execute the 1st OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clFinish(clCommandQue);
	
	globalWorkSize[0] = (size_t)ceil(((float)NY) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = 1;

	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&r_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void *)&s_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 3, sizeof(int), &nx);
        errcode |= clSetKernelArg(clKernel2, 4, sizeof(int), &ny);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");
	
	// Execute the 2nd OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clFinish(clCommandQue);

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;
}


void cl_clean_up()
{
	// Clean up
	errcode = clFlush(clCommandQue);
	errcode = clFinish(clCommandQue);
	errcode = clReleaseKernel(clKernel1);
	errcode = clReleaseKernel(clKernel2);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(p_mem_obj);
	errcode = clReleaseMemObject(q_mem_obj);
	errcode = clReleaseMemObject(r_mem_obj);
	errcode = clReleaseMemObject(s_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


void bicg_cpu(int nx, int ny, DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny), DATA_TYPE POLYBENCH_1D(r,NX,nx), DATA_TYPE POLYBENCH_1D(s,NY,ny), 
		DATA_TYPE POLYBENCH_1D(p,NY,ny), DATA_TYPE POLYBENCH_1D(q,NX,nx))
{
	int i,j;
	
  	for (i = 0; i < _PB_NY; i++)
	{
		s[i] = 0.0;
	}

	for (i = 0; i < _PB_NX; i++)
	{
		q[i] = 0.0;
		for (j = 0; j < _PB_NY; j++)
	  	{
	    		s[j] = s[j] + r[i] * A[i][j];
	    		q[i] = q[i] + A[i][j] * p[j];
	  	}
	}
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nx, int ny,
		 DATA_TYPE POLYBENCH_1D(s,NY,ny),
		 DATA_TYPE POLYBENCH_1D(q,NX,nx))

{
  int i;

  for (i = 0; i < ny; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, s[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
  for (i = 0; i < nx; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, q[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
  fprintf (stderr, "\n");
}


int main(int argc, char *argv[])
{
	int nx = NX;
	int ny = NY;

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NX,NY,nx,ny);
	POLYBENCH_1D_ARRAY_DECL(s,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(q,DATA_TYPE,NX,nx);
	POLYBENCH_1D_ARRAY_DECL(p,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(r,DATA_TYPE,NX,nx);
	POLYBENCH_1D_ARRAY_DECL(s_outputFromGpu,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(q_outputFromGpu,DATA_TYPE,NX,nx);

	init_array(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(r));
	
	read_cl_file();
	cl_initialization();
	cl_mem_init(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(r), POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q));
	cl_load_prog();

	cl_launch_kernel(nx, ny);

	errcode = clEnqueueReadBuffer(clCommandQue, s_mem_obj, CL_TRUE, 0, NY*sizeof(DATA_TYPE), POLYBENCH_ARRAY(s_outputFromGpu), 0, NULL, NULL);
	errcode = clEnqueueReadBuffer(clCommandQue, q_mem_obj, CL_TRUE, 0, NX*sizeof(DATA_TYPE), POLYBENCH_ARRAY(q_outputFromGpu), 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");  

	#if RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		bicg_cpu(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(r), POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q));
	
		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;

		compareResults(nx, ny, POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(s_outputFromGpu), POLYBENCH_ARRAY(q), 
			POLYBENCH_ARRAY(q_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(nx, ny, POLYBENCH_ARRAY(s_outputFromGpu), POLYBENCH_ARRAY(q_outputFromGpu)));
	
	#endif //RUN_ON_CPU

	cl_clean_up();
	
	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(r);
	POLYBENCH_FREE_ARRAY(s);
	POLYBENCH_FREE_ARRAY(p);
	POLYBENCH_FREE_ARRAY(q);
	POLYBENCH_FREE_ARRAY(s_outputFromGpu);
	POLYBENCH_FREE_ARRAY(q_outputFromGpu);
	
    	return 0;
}

#include <polybench.c>
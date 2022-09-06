/**
 * gemver.c: This file is part of the PolyBench/GPU 1.0 test suite.
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

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define POLYBENCH_TIME 1


#include "gemver.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

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
cl_mem u1_mem_obj;
cl_mem v1_mem_obj;
cl_mem u2_mem_obj;
cl_mem v2_mem_obj;
cl_mem w_mem_obj;
cl_mem x_mem_obj;
cl_mem y_mem_obj;
cl_mem z_mem_obj;


FILE *fp;
char *source_str;
size_t source_size;



void compareResults(int n, DATA_TYPE POLYBENCH_1D(w1, N, n), DATA_TYPE POLYBENCH_1D(w2, N, n))
{
	int i, fail;
	fail = 0;
	
	for (i=0; i < N; i++) 
	{
		if (percentDiff(w1[i], w2[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
		{
			fail++;
		}
	}
		
	// Print results
	printf("Number of misses: %d\n", fail);
}


void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen("gemver.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init(int n, DATA_TYPE *alpha,
	DATA_TYPE *beta,
	DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
	DATA_TYPE POLYBENCH_1D(u1,N,n),
	DATA_TYPE POLYBENCH_1D(v1,N,n),
	DATA_TYPE POLYBENCH_1D(u2,N,n),
	DATA_TYPE POLYBENCH_1D(v2,N,n),
	DATA_TYPE POLYBENCH_1D(w,N,n),
	DATA_TYPE POLYBENCH_1D(x,N,n),
	DATA_TYPE POLYBENCH_1D(y,N,n),
	DATA_TYPE POLYBENCH_1D(z,N,n))
{
	int i, j;

	*alpha = 43532;
	*beta = 12313;

  	for (i = 0; i < N; i++)
	{
	    	u1[i] = i;
	    	u2[i] = (i+1)/N/2.0;
	    	v1[i] = (i+1)/N/4.0;
	    	v2[i] = (i+1)/N/6.0;
	    	y[i] = (i+1)/N/8.0;
	    	z[i] = (i+1)/N/9.0;
	    	x[i] = 0.0;
	    	w[i] = 0.0;

    		for (j = 0; j < N; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / N;
		}
	}
}


void cl_initialization()
{
	
	// Get platform and device information
	errcode = clGetPlatformIDs(1, &platform_id, &num_platforms);
	if(errcode == CL_SUCCESS) printf("number of platforms is %d\n",num_platforms);

	errcode = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform name is %s\n",str_temp);

	errcode = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform version is %s\n",str_temp);

	errcode = clGetDeviceIDs( platform_id, OPENCL_DEVICE_SELECTION, 1, &device_id, &num_devices);
	if(errcode == CL_SUCCESS) printf("device id is %d\n",device_id);

	errcode = clGetDeviceInfo(device_id,CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("device name is %s\n",str_temp);
	
	// Create an OpenCL context
	clGPUContext = clCreateContext( NULL, 1, &device_id, NULL, NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating context\n");
 
	//Create a command-queue
	clCommandQue = clCreateCommandQueue(clGPUContext, device_id, 0, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating command queue\n");
}


void cl_mem_init(DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_1D(u1,N,n), DATA_TYPE POLYBENCH_1D(v1,N,n), DATA_TYPE POLYBENCH_1D(u2,N,n), 
	DATA_TYPE POLYBENCH_1D(v2,N,n), DATA_TYPE POLYBENCH_1D(w,N,n), DATA_TYPE POLYBENCH_1D(x,N,n), DATA_TYPE POLYBENCH_1D(y,N,n), 
	DATA_TYPE POLYBENCH_1D(z,N,n))
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N * N, NULL, &errcode);
	u1_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	v1_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	u2_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	v2_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	w_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	x_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	y_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	z_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N * N, A, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, u1_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, u1, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, v1_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, v1, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, u2_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, u2, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, v2_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, v2, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, w_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, w, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, x_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, x, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, y_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, y, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, z_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, z, 0, NULL, NULL);
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
		
	// Create the OpenCL kernel
	clKernel1 = clCreateKernel(clProgram, "gemver_kernel1", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");

	// Create the OpenCL kernel
	clKernel2 = clCreateKernel(clProgram, "gemver_kernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel2\n");

	// Create the OpenCL kernel
	clKernel3 = clCreateKernel(clProgram, "gemver_kernel3", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel3\n");
	clFinish(clCommandQue);
}


void cl_launch_kernel(int n, DATA_TYPE alpha, DATA_TYPE beta)
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_KERNEL_1_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_KERNEL_1_Y;
	globalWorkSize[0] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_1_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_1_X;
	globalWorkSize[1] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_1_Y)) * DIM_LOCAL_WORK_GROUP_KERNEL_1_Y;
	
	/* Start timer. */
  	polybench_start_instruments;

	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&v1_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&v2_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 3, sizeof(cl_mem), (void *)&u1_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 4, sizeof(cl_mem), (void *)&u2_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 5, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");
	size_t global_item_size = sizeof(DATA_TYPE) * N; 
	size_t local_item_size = 64; 
	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel1\n");
	clEnqueueBarrier(clCommandQue);

	int dim = N;
	
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_KERNEL_2_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_KERNEL_2_Y;
	globalWorkSize[0] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_2_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_2_X;
	globalWorkSize[1] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_2_Y)) * DIM_LOCAL_WORK_GROUP_KERNEL_2_Y;

	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&x_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void *)&y_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 3, sizeof(cl_mem), (void *)&z_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 4, sizeof(DATA_TYPE), (void *)&beta);
	errcode |= clSetKernelArg(clKernel2, 5, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments2\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel2\n");
	clEnqueueBarrier(clCommandQue);
	
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_KERNEL_3_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_KERNEL_3_Y;
	globalWorkSize[0] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_3_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_3_X;
	globalWorkSize[1] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_3_Y)) * DIM_LOCAL_WORK_GROUP_KERNEL_3_Y;

	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel3, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel3, 1, sizeof(cl_mem), (void *)&x_mem_obj);
	errcode |= clSetKernelArg(clKernel3, 2, sizeof(cl_mem), (void *)&w_mem_obj);
	errcode |= clSetKernelArg(clKernel3, 3, sizeof(DATA_TYPE), (void *)&alpha);
	errcode |= clSetKernelArg(clKernel3, 4, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments3\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel3, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel3\n");
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
	errcode = clReleaseKernel(clKernel3);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(u1_mem_obj);
	errcode = clReleaseMemObject(v1_mem_obj);
	errcode = clReleaseMemObject(u2_mem_obj);
	errcode = clReleaseMemObject(v2_mem_obj);
	errcode = clReleaseMemObject(w_mem_obj);
	errcode = clReleaseMemObject(x_mem_obj);
	errcode = clReleaseMemObject(y_mem_obj);
	errcode = clReleaseMemObject(z_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


void gemver(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_1D(u1, N, n), DATA_TYPE POLYBENCH_1D(v1, N, n), 
	DATA_TYPE POLYBENCH_1D(u2, N, n), DATA_TYPE POLYBENCH_1D(v2, N, n), DATA_TYPE POLYBENCH_1D(w, N, n), DATA_TYPE POLYBENCH_1D(x, N, n), DATA_TYPE POLYBENCH_1D(y, N, n), 
	DATA_TYPE POLYBENCH_1D(z, N, n))
{
	int i,j;
	
  	for (i = 0; i < _PB_N; i++)
	{
    		for (j = 0; j < _PB_N; j++)
		{
      			A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
		}
	}

  	for (i = 0; i < _PB_N; i++)
	{
    		for (j = 0; j < _PB_N; j++)
		{
      			x[i] = x[i] + beta * A[j][i] * y[j];
		}
	}

  	for (i = 0; i < _PB_N; i++)
	{
    		x[i] = x[i] + z[i];
	}

  	for (i = 0; i < _PB_N; i++)
	{
    		for (j = 0; j < _PB_N; j++)
		{
      			w[i] = w[i] +  alpha * A[i][j] * x[j];
		}
	}
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(w,N,n))
{
  int i;

  for (i = 0; i < n; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, w[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
}


int main(int argc, char *argv[])
{
	/* Retrieve problem size. */
	int n = N;

	/* Variable declaration/allocation. */
	DATA_TYPE alpha;
	DATA_TYPE beta;

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,N,n,n);
	POLYBENCH_1D_ARRAY_DECL(u1,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(v1,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(u2,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(v2,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(w,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(w_outputFromGpu,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(x,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(y,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(z,DATA_TYPE,N,n);
	
	init(n, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(u1),
	      POLYBENCH_ARRAY(v1),
	      POLYBENCH_ARRAY(u2),
	      POLYBENCH_ARRAY(v2),
	      POLYBENCH_ARRAY(w),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y),
	      POLYBENCH_ARRAY(z));

	read_cl_file();
	cl_initialization();
	cl_mem_init(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(u1), POLYBENCH_ARRAY(v1), POLYBENCH_ARRAY(u2), POLYBENCH_ARRAY(v2),
	      POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(z));

	cl_load_prog();
	cl_launch_kernel(n, alpha, beta);
	errcode = clEnqueueReadBuffer(clCommandQue, w_mem_obj, CL_TRUE, 0, N*sizeof(DATA_TYPE), POLYBENCH_ARRAY(w_outputFromGpu), 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");

	#if RUN_ON_CPU
	
		/* Start timer. */
	  	polybench_start_instruments;

		gemver(n, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(u1), POLYBENCH_ARRAY(v1), POLYBENCH_ARRAY(u2), POLYBENCH_ARRAY(v2), 
			POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(z));
	
		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
  		polybench_stop_instruments;
 		polybench_print_instruments;

		compareResults(n, POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(w_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w_outputFromGpu)));

	#endif //RUN_ON_CPU


	cl_clean_up();
	
	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(u1);
	POLYBENCH_FREE_ARRAY(v1);
	POLYBENCH_FREE_ARRAY(u2);
	POLYBENCH_FREE_ARRAY(v2);
	POLYBENCH_FREE_ARRAY(w);  
	POLYBENCH_FREE_ARRAY(w_outputFromGpu);  
	POLYBENCH_FREE_ARRAY(x);  
	POLYBENCH_FREE_ARRAY(y);
	POLYBENCH_FREE_ARRAY(z);
	
	
    return 0;
}

#include <polybench.c>
/*********************************************************************************/
//
// Polybench kernels implementation on OpenCL GPU
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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define MAX_SOURCE_SIZE (0x100000)

/* Problem size. */
#define N 4096

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

typedef float DATA_TYPE;

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
FILE *fp;
char *source_str;
size_t source_size;



void compareResults(DATA_TYPE* A_cpu, DATA_TYPE* A_outputFromGpu)
{
	int i, j, fail;
	fail = 0;
	
	// Compare a and b
	for (i=2; i<N-2; i++) 
	{
		for (j=2; j<(N-2); j++) 
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


void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen("lu.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init_array(DATA_TYPE* A)
{
	int i, j;

	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			A[i*N + j] = ((DATA_TYPE) i*j + 1.0f) / (float)N;
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

	errcode = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
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


void cl_mem_init(DATA_TYPE* A)
{
	size_t mem_size_A = N*N*sizeof(DATA_TYPE);

	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, mem_size_A, NULL, &errcode);
			
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, mem_size_A, A, 0, NULL, NULL);
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
	clKernel1 = clCreateKernel(clProgram, "lu_kernel1", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");
	clKernel2 = clCreateKernel(clProgram, "lu_kernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel2\n");
	clFinish(clCommandQue);
}


void cl_launch_kernel1(int k)
{
	if (k < (N-1))
	{
		size_t localWorkSize[2], globalWorkSize[2];
		localWorkSize[0] = 256;
		localWorkSize[1] = 1;
		globalWorkSize[0] = (size_t)ceil(((double)N - (double)(k + 1)) / 256.0) * 256;
		globalWorkSize[1] = 1;
	
		// Set the arguments of the kernel
		errcode = clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
		errcode |= clSetKernelArg(clKernel1, 1, sizeof(int), (void *)&k);
	
		if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
		clFinish(clCommandQue);
	}
}


void cl_launch_kernel2(int k)
{
	if (k < (N-1))
	{
		size_t localWorkSize[2], globalWorkSize[2];
		localWorkSize[0] = 32;
		localWorkSize[1] = 8;
		globalWorkSize[0] = (size_t)ceil(((double)N - (double)(k + 1)) / 32.0) * 32;
		globalWorkSize[1] = (size_t)ceil(((double)N - (double)(k + 1)) / 8.0) * 8;
	
		// Set the arguments of the kernel
		errcode = clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&a_mem_obj);
		errcode |= clSetKernelArg(clKernel2, 1, sizeof(int), (void *)&k);
	
		if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) 
		{
			printf("Error in launching kernel\n");
			printf("Nums: %d %d\n", globalWorkSize[0], globalWorkSize[1]);
		}
		clFinish(clCommandQue);
	}
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
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


void lu(DATA_TYPE* A)
{
	int k, j, i;
	for (k = 0; k < N; k++)
    	{
		for (j = k + 1; j < N; j++)
		{
			A[k*N + j] = A[k*N + j] / A[k*N + k];
		}

		for(i = k + 1; i < N; i++)
		{
			int j;
			for (j = k + 1; j < N; j++)
			{
				A[i*N + j] = A[i*N + j] - A[i*N + k] * A[k*N + j];
			}
		}
    	}
}


int main(void) 
{
	double t_start, t_end;
	int i;

	DATA_TYPE* A;
	DATA_TYPE* A_outputFromGpu;

	A = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));
	A_outputFromGpu = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));

	init_array(A);

	read_cl_file();
	cl_initialization();
	cl_mem_init(A);
	cl_load_prog();
	t_start = rtclock();
	
	int k;
	for (k = 0; k < N; k++)
    	{
		cl_launch_kernel1(k);
		cl_launch_kernel2(k);
	}

	t_end = rtclock();
	errcode = clEnqueueReadBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, N*N*sizeof(DATA_TYPE), A_outputFromGpu, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");
	
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	t_start = rtclock();
	lu(A);
	t_end = rtclock(); 
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);   
	compareResults(A, A_outputFromGpu);
	cl_clean_up();

	free(A);
	free(A_outputFromGpu);

    	return 0;
}


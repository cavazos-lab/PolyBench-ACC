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

#define GPU_DEVICE 0

#define MAX_SOURCE_SIZE (0x10000000)

/* Problem size. */
#define TSTEPS 5
#define N 1024

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 256
#define DIM_LOCAL_WORK_GROUP_Y 1

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

typedef float real;

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
cl_kernel clKernel4;
cl_kernel clKernel5;
cl_kernel clKernel6;
cl_command_queue clCommandQue;
cl_program clProgram;
cl_mem a_mem_obj;
cl_mem b_mem_obj;
cl_mem c_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;
unsigned int mem_size_A;
unsigned int mem_size_B;
unsigned int mem_size_C;

void init_array(real* A, real* B, real* X)
{
  int i, j;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      {
	X[i*N + j] = ((real) i*(j+1) + 1) / N;
	A[i*N + j] = ((real) (i-1)*(j+4) + 2) / N;
	B[i*N + j] = ((real) (i+3)*(j+7) + 3) / N;
      }
}


void compareResults(real* B1, real* B2, real* X1, real* X2){
	int i, j, fail;
	fail = 0;
	
	// Compare a and b
	for (i=0; i<N; i++) 
	{
		for (j=0; j<(N); j++) 
		{
			if (percentDiff(B1[i*N + j], B2[i*N + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	for (i=0; i<N; i++) 
	{
		for (j=0; j<(N); j++)
		{
			if (percentDiff(X1[i*N + j], X2[i*N + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
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
	fp = fopen("adi.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
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

void cl_mem_init(real* A, real* B, real* X)
{
	mem_size_A = N*N*sizeof(real);
	mem_size_B = N*N*sizeof(real);
	mem_size_C = N*N*sizeof(real);

	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, mem_size_A, NULL, &errcode);
	b_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, mem_size_B, NULL, &errcode);
	c_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, mem_size_C, NULL, &errcode);
		
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, mem_size_A, A, 0, NULL, NULL);
	errcode |= clEnqueueWriteBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, mem_size_B, B, 0, NULL, NULL);
	errcode |= clEnqueueWriteBuffer(clCommandQue, c_mem_obj, CL_TRUE, 0, mem_size_C, X, 0, NULL, NULL);
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
	clKernel1 = clCreateKernel(clProgram, "adi_kernel1", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");
	clKernel2 = clCreateKernel(clProgram, "adi_kernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel2\n");
	clKernel3 = clCreateKernel(clProgram, "adi_kernel3", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel3\n");
	clKernel4 = clCreateKernel(clProgram, "adi_kernel4", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel4\n");
	clKernel5 = clCreateKernel(clProgram, "adi_kernel5", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel5\n");
	clKernel6 = clCreateKernel(clProgram, "adi_kernel6", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel6\n");
	clFinish(clCommandQue);
}

void cl_launch_kernel1()
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = 1;
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&c_mem_obj);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clFinish(clCommandQue);
}

void cl_launch_kernel2()
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = 1;
	globalWorkSize[0] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = 1;
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void *)&c_mem_obj);
	//if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	//if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clFinish(clCommandQue);
}

void cl_launch_kernel3()
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = 1;
	globalWorkSize[0] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = 1;
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel3, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel3, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	errcode |= clSetKernelArg(clKernel3, 2, sizeof(cl_mem), (void *)&c_mem_obj);
	//if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel3, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	//if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clFinish(clCommandQue);
}

void cl_launch_kernel4(int i)
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = 1;
	globalWorkSize[0] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = 1;
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel4, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel4, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	errcode |= clSetKernelArg(clKernel4, 2, sizeof(cl_mem), (void *)&c_mem_obj);
	errcode |= clSetKernelArg(clKernel4, 3, sizeof(int), (void *)&i);
	//if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel4, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	//if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clFinish(clCommandQue);
}

void cl_launch_kernel5()
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = 1;
	globalWorkSize[0] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = 1;
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel5, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel5, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	errcode |= clSetKernelArg(clKernel5, 2, sizeof(cl_mem), (void *)&c_mem_obj);
	//if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel5, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	//if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clFinish(clCommandQue);
}

void cl_launch_kernel6(int i)
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = 1;
	globalWorkSize[0] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = 1;
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel6, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel6, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	errcode |= clSetKernelArg(clKernel6, 2, sizeof(cl_mem), (void *)&c_mem_obj);
	errcode |= clSetKernelArg(clKernel6, 3, sizeof(int), (void *)&i);
	//if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel6, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	//if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
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
	errcode = clReleaseKernel(clKernel4);
	errcode = clReleaseKernel(clKernel5);
	errcode = clReleaseKernel(clKernel6);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(b_mem_obj);
	errcode = clReleaseMemObject(c_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}

void adi(real* A, real* B, real* X)
{
	int t;
	int i1;
	int i2;

	for (t = 0; t < TSTEPS; t++)
    	{
      		for (i1 = 0; i1 < N; i1++)
			for (i2 = 1; i2 < N; i2++)
	  		{
	    			X[i1*N + i2] = X[i1*N + i2] - X[i1*N + (i2-1)] * A[i1*N + i2] / B[i1*N + (i2-1)];
	    			B[i1*N + i2] = B[i1*N + i2] - A[i1*N + i2] * A[i1*N + i2] / B[i1*N + (i2-1)];
	  		}

      		for (i1 = 0; i1 < N; i1++)
			X[i1*N + (N-1)] = X[i1*N + (N-1)] / B[i1*N + (N-1)];

      		for (i1 = 0; i1 < N; i1++)
			for (i2 = 0; i2 < N-2; i2++)
	  			X[i1*N + (N-i2-2)] = (X[i1*N + (N-2-i2)] - X[i1*N + (N-2-i2-1)] * A[i1*N + (N-i2-3)]) / B[i1*N + (N-3-i2)];

      		for (i1 = 1; i1 < N; i1++)
			for (i2 = 0; i2 < N; i2++) 
			{
	  			X[i1*N + i2] = X[i1*N + i2] - X[(i1-1)*N + i2] * A[i1*N + i2] / B[(i1-1)*N + i2];
	  			B[i1*N + i2] = B[i1*N + i2] - A[i1*N + i2] * A[i1*N + i2] / B[(i1-1)*N + i2];
			}

      		for (i2 = 0; i2 < N; i2++)
			X[(N-1)*N + i2] = X[(N-1)*N + i2] / B[(N-1)*N + i2];

      		for (i1 = 0; i1 < N-2; i1++)
			for (i2 = 0; i2 < N; i2++)
	  			X[(N-2-i1)*N + i2] = (X[(N-2-i1)*N + i2] - X[(N-i1-3)*N + i2] * A[(N-3-i1)*N + i2]) / B[(N-2-i1)*N + i2];
    }
}


int main(void) {
	double t_start, t_end;
	int i;

	real* A1;
	real* A2;
	real* B1;
	real* B2;
	real* X1;
	real* X2;

	A1 = (real*)malloc(N*N*sizeof(real));
	A2 = (real*)malloc(N*N*sizeof(real));
	B1 = (real*)malloc(N*N*sizeof(real));
	B2 = (real*)malloc(N*N*sizeof(real));
	X1 = (real*)malloc(N*N*sizeof(real));
	X2 = (real*)malloc(N*N*sizeof(real));

	init_array(A1, B1, X1);
	init_array(A2, B2, X2);

	read_cl_file();
	cl_initialization();
	cl_mem_init(A1, B1, X1);
	cl_load_prog();
	t_start = rtclock();

	int t, i1;

	for (t = 0; t < TSTEPS; t++)
	{
		cl_launch_kernel1();

		cl_launch_kernel2();

		cl_launch_kernel3();
	
		for (i1 = 1; i1 < N; i1++)
		{
			cl_launch_kernel4(i1);
		}

		cl_launch_kernel5();
		
		for (i1 = 0; i1 < N-2; i1++)
		{
			cl_launch_kernel6(i1);
		}
	}	
	
	t_end = rtclock();

	errcode = clEnqueueReadBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, mem_size_B, B1, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");
	errcode = clEnqueueReadBuffer(clCommandQue, c_mem_obj, CL_TRUE, 0, mem_size_C, X1, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");
	
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	t_start = rtclock();
	adi(A2, B2, X2);
	t_end = rtclock(); 
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);   
	compareResults(B1, B2, X1, X2);
	cl_clean_up();
	free(A1);
	free(A2);
	free(B1);
	free(B2);
	free(X1);
	free(X2);
    	return 0;
}

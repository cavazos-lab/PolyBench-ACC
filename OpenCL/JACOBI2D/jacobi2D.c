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

#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

/* Problem size. */
#define TSTEPS 100
#define N 4096

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Can switch DATA_TYPE between float and double */
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
cl_mem b_mem_obj;
cl_mem c_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;



void compareResults(DATA_TYPE* a1, DATA_TYPE* a2, DATA_TYPE* b1, DATA_TYPE* b2)
{
	int i, j, fail;
	fail = 0;   

	// Compare a and c
	for (i=0; i<N; i++) 
	{
      	for (j=0; j<N; j++) 
		{
			if (percentDiff(a1[i*N + j], a2[i*N + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}

	for (i=0; i<N; i++) 
	{
       	for (j=0; j<N; j++) 
		{
           	if (percentDiff(b1[i*N + j], b2[i*N + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
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
	fp = fopen("jacobi2D.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init_array(DATA_TYPE* A, DATA_TYPE* B)
{
  int i, j;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      {
	A[i*N + j] = ((DATA_TYPE) i*(j+2) + 10) / N;
	B[i*N + j] = ((DATA_TYPE) (i-4)*(j-1) + 11) / N;
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


void cl_mem_init(DATA_TYPE* A, DATA_TYPE* B)
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, N * N * sizeof(DATA_TYPE), NULL, &errcode);
	b_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, N * N * sizeof(DATA_TYPE), NULL, &errcode);
		
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, N * N * sizeof(DATA_TYPE), A, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, N * N * sizeof(DATA_TYPE), B, 0, NULL, NULL);
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
	clKernel1 = clCreateKernel(clProgram, "runJacobi2D_kernel1", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");
	clKernel2 = clCreateKernel(clProgram, "runJacobi2D_kernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel2\n");
	clFinish(clCommandQue);
}

void cl_launch_kernel1()
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = N;
	globalWorkSize[1] = N;
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clFinish(clCommandQue);
}

void cl_launch_kernel2()
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = N;
	globalWorkSize[1] = N;
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clFinish(clCommandQue);
}

void cl_launch_kernels()
{

	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = N;
	globalWorkSize[1] = N;
	int t;
	
	for (t = 0; t < TSTEPS ; t++)
	{	
		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
		clFinish(clCommandQue);

		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
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
	errcode = clReleaseMemObject(b_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


void jacobi2D(DATA_TYPE* A, DATA_TYPE* B)
{
	int t, i, j;
	//double t_start, t_end;

	for (t = 0; t < TSTEPS; t++)
    	{
      		for (i = 2; i < N - 1; i++)
			for (j = 2; j < N - 1; j++)
	  			B[i*N + j] = 0.2f * (A[i*N + j] + A[i*N + (j-1)] + A[i*N + (1+j)] + A[(1+i)*N + j] + A[(i-1)*N + j]);

      		for (i = 2; i < N-1; i++)
			for (j = 2; j < N-1; j++)
	  			A[i*N + j] = B[i*N + j];
    	}


	//fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
}


int main(void) {

	DATA_TYPE* a1;
	DATA_TYPE* b1;
	DATA_TYPE* a2;
	DATA_TYPE* b2;

	a1 = (DATA_TYPE*)malloc(N * N * sizeof(DATA_TYPE));
	b1 = (DATA_TYPE*)malloc(N * N * sizeof(DATA_TYPE));

	a2 = (DATA_TYPE*)malloc(N * N * sizeof(DATA_TYPE));
	b2 = (DATA_TYPE*)malloc(N * N * sizeof(DATA_TYPE));

	init_array(a1, b1);
	init_array(a2, b2);

	double t_start, t_end;
	int i;
	
	read_cl_file();
	cl_initialization();
	cl_mem_init(a1, b1);
	cl_load_prog();
	t_start = rtclock();


	int t;
	for (t = 0; t < TSTEPS ; t++)
    	{
		cl_launch_kernel1();
		cl_launch_kernel2();
	}



	t_end = rtclock();
	
	errcode = clEnqueueReadBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, N * N * sizeof(DATA_TYPE), a1, 0, NULL, NULL);
	errcode = clEnqueueReadBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, N * N * sizeof(DATA_TYPE), b1, 0, NULL, NULL);
	
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");
	
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	t_start = rtclock();
	jacobi2D(a2, b2);
	t_end = rtclock(); 
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);   
	compareResults(a1, a2, b1, b2);
	cl_clean_up();
	free(a1);
	free(a2);
	free(b1);
	free(b2);
    return 0;
}

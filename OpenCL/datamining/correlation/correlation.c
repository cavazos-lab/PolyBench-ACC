/**
 * correlation.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#include <math.h>
#include <sys/time.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define POLYBENCH_TIME 1


#include "correlation.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define MAX_SOURCE_SIZE (0x100000)


#define sqrt_of_array_cell(x,j) sqrt(x[j])

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

char str_temp[1024];

#define FLOAT_N 3214212.01
#define EPS 0.005

cl_platform_id platform_id;
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel_mean;
cl_kernel clKernel_std;
cl_kernel clKernel_reduce;
cl_kernel clKernel_corr;
cl_command_queue clCommandQue;
cl_program clProgram;
cl_mem data_mem_obj;
cl_mem stddev_mem_obj;
cl_mem mean_mem_obj;
cl_mem symmat_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;



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


void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen("correlation.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


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


void cl_mem_init(DATA_TYPE POLYBENCH_2D(data,M,N,m,n), DATA_TYPE POLYBENCH_1D(mean,M,m), DATA_TYPE POLYBENCH_1D(stddev,M,m), DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m))
{
	data_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M * N, NULL, &errcode);
	symmat_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M * N, NULL, &errcode);
	stddev_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M, NULL, &errcode);
	mean_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M, NULL, &errcode);
		
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, data_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * M * N, data, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, symmat_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * M * N, symmat, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, stddev_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * M, stddev, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, mean_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * M, mean, 0, NULL, NULL);
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
	clKernel_mean = clCreateKernel(clProgram, "mean_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");

	clKernel_std = clCreateKernel(clProgram, "std_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel2\n");

	clKernel_reduce = clCreateKernel(clProgram, "reduce_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel3\n");

	clKernel_corr = clCreateKernel(clProgram, "corr_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel4\n");
	clFinish(clCommandQue);
}


void cl_launch_kernel(int m, int n)
{
	DATA_TYPE float_n = FLOAT_N;
	DATA_TYPE eps = EPS;

	size_t localWorkSize_Kernel1[2], globalWorkSize_Kernel1[2];
	size_t localWorkSize_Kernel2[2], globalWorkSize_Kernel2[2];
	size_t localWorkSize_Kernel3[2], globalWorkSize_Kernel3[2];
	size_t localWorkSize_Kernel4[2], globalWorkSize_Kernel4[2];

	localWorkSize_Kernel1[0] = DIM_LOCAL_WORK_GROUP_KERNEL_1_X;
	localWorkSize_Kernel1[1] = DIM_LOCAL_WORK_GROUP_KERNEL_1_Y;
	globalWorkSize_Kernel1[0] = (size_t)ceil(((float)M) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_1_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_1_X;
	globalWorkSize_Kernel1[1] = 1;

	localWorkSize_Kernel2[0] = DIM_LOCAL_WORK_GROUP_KERNEL_2_X;
	localWorkSize_Kernel2[1] = DIM_LOCAL_WORK_GROUP_KERNEL_2_Y;
	globalWorkSize_Kernel2[0] = (size_t)ceil(((float)M) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_2_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_2_X;
	globalWorkSize_Kernel2[1] = 1;

	localWorkSize_Kernel3[0] = DIM_LOCAL_WORK_GROUP_KERNEL_3_X;
	localWorkSize_Kernel3[1] = DIM_LOCAL_WORK_GROUP_KERNEL_3_Y;
	globalWorkSize_Kernel3[0] = (size_t)ceil(((float)M) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_3_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_3_X;
	globalWorkSize_Kernel3[1] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_3_Y)) * DIM_LOCAL_WORK_GROUP_KERNEL_3_Y;

	localWorkSize_Kernel4[0] = DIM_LOCAL_WORK_GROUP_KERNEL_4_X;
	localWorkSize_Kernel4[1] = DIM_LOCAL_WORK_GROUP_KERNEL_4_Y;
	globalWorkSize_Kernel4[0] = (size_t)ceil(((float)M) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_4_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_4_X;
	globalWorkSize_Kernel4[1] = 1;

	/* Start timer. */
  	polybench_start_instruments;
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel_mean, 0, sizeof(cl_mem), (void *)&mean_mem_obj);
	errcode |= clSetKernelArg(clKernel_mean, 1, sizeof(cl_mem), (void *)&data_mem_obj);
	errcode |= clSetKernelArg(clKernel_mean, 2, sizeof(DATA_TYPE), (void *)&float_n);
	errcode |= clSetKernelArg(clKernel_mean, 3, sizeof(int), (void *)&m);
	errcode |= clSetKernelArg(clKernel_mean, 4, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel_mean, 1, NULL, globalWorkSize_Kernel1, localWorkSize_Kernel1, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel1\n");
	clEnqueueBarrier(clCommandQue);

	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel_std, 0, sizeof(cl_mem), (void *)&mean_mem_obj);
	errcode =  clSetKernelArg(clKernel_std, 1, sizeof(cl_mem), (void *)&stddev_mem_obj);
	errcode |= clSetKernelArg(clKernel_std, 2, sizeof(cl_mem), (void *)&data_mem_obj);
	errcode |= clSetKernelArg(clKernel_std, 3, sizeof(DATA_TYPE), (void *)&float_n);
	errcode |= clSetKernelArg(clKernel_std, 4, sizeof(DATA_TYPE), (void *)&eps);
	errcode |= clSetKernelArg(clKernel_std, 5, sizeof(int), (void *)&m);
	errcode |= clSetKernelArg(clKernel_std, 6, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments2\n");
 
	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel_std, 1, NULL, globalWorkSize_Kernel2, localWorkSize_Kernel2, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel2\n");
	clEnqueueBarrier(clCommandQue);

	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel_reduce, 0, sizeof(cl_mem), (void *)&mean_mem_obj);
	errcode =  clSetKernelArg(clKernel_reduce, 1, sizeof(cl_mem), (void *)&stddev_mem_obj);
	errcode |= clSetKernelArg(clKernel_reduce, 2, sizeof(cl_mem), (void *)&data_mem_obj);
	errcode |= clSetKernelArg(clKernel_reduce, 3, sizeof(DATA_TYPE), (void *)&float_n);
	errcode |= clSetKernelArg(clKernel_reduce, 4, sizeof(int), (void *)&m);
	errcode |= clSetKernelArg(clKernel_reduce, 5, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments3\n");
 
	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel_reduce, 2, NULL, globalWorkSize_Kernel3, localWorkSize_Kernel3, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel3\n");
	clEnqueueBarrier(clCommandQue);

	// Set the arguments of the kernel	
	errcode =  clSetKernelArg(clKernel_corr, 0, sizeof(cl_mem), (void *)&symmat_mem_obj);
	errcode |= clSetKernelArg(clKernel_corr, 1, sizeof(cl_mem), (void *)&data_mem_obj);
	errcode |= clSetKernelArg(clKernel_corr, 2, sizeof(int), (void *)&m);
	errcode |= clSetKernelArg(clKernel_corr, 3, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments4\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel_corr, 1, NULL, globalWorkSize_Kernel4, localWorkSize_Kernel4, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel4\n");
	clEnqueueBarrier(clCommandQue);

	DATA_TYPE val = 1.0;
	clEnqueueWriteBuffer(clCommandQue, symmat_mem_obj, CL_TRUE, ((M-1)*M + (M-1))*sizeof(DATA_TYPE), sizeof(DATA_TYPE), &val, 0, NULL, NULL);

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
	errcode = clReleaseKernel(clKernel_reduce);
	errcode = clReleaseKernel(clKernel_mean);
	errcode = clReleaseKernel(clKernel_std);
	errcode = clReleaseKernel(clKernel_corr);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(symmat_mem_obj);
	errcode = clReleaseMemObject(data_mem_obj);
	errcode = clReleaseMemObject(mean_mem_obj);
	errcode = clReleaseMemObject(stddev_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
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


int main(int argc, char *argv[])
{	
	int m = M;
	int n = N;

	POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,M,N,m,n);
  	POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
  	POLYBENCH_1D_ARRAY_DECL(stddev,DATA_TYPE,M,m);
	POLYBENCH_2D_ARRAY_DECL(symmat,DATA_TYPE,M,N,m,n);
  	POLYBENCH_2D_ARRAY_DECL(symmat_outputFromGpu,DATA_TYPE,M,N,m,n);
  	
	init_arrays(m, n, POLYBENCH_ARRAY(data));

	read_cl_file();
	cl_initialization();
	cl_mem_init(POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(stddev), POLYBENCH_ARRAY(symmat_outputFromGpu));
	cl_load_prog();

	cl_launch_kernel(m, n);

	errcode = clEnqueueReadBuffer(clCommandQue, symmat_mem_obj, CL_TRUE, 0, M * N * sizeof(DATA_TYPE), POLYBENCH_ARRAY(symmat_outputFromGpu), 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");

	#if RUN_ON_CPU

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

	cl_clean_up();
	
	return 0;
}

#include <polybench.c>
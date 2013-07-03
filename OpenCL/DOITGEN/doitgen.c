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
#define NR 512
#define NQ 512
#define NP 512

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Can switch DATA_TYPE between float and double */
typedef double DATA_TYPE;

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






void compareResults(DATA_TYPE* sum1, DATA_TYPE* sum2){

	int fail = 0;

	int r, q, p;
	
	for (r = 0; r < NR; r++)
	{
    		for (q = 0; q < NQ; q++)  
		{
      			for (p = 0; p < NP; p++)  
			{
				if (percentDiff(sum1[r * (NQ * NP) + q * NP + p], sum2[r * (NQ * NP) + q * NP + p]) > PERCENT_DIFF_ERROR_THRESHOLD)
				{
					fail++;
				}
			}
		}
	}
	
	// Print results
	printf("Number of misses: %d\n", fail);
}


void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen("doitgen.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}

void init_array(DATA_TYPE *A, DATA_TYPE *C4)
{
	int i, j, k;
  	
	for (i = 0; i < NR; i++)
  	{
    	for (j = 0; j < NQ; j++)
    	{
      		for (k = 0; k < NP; k++)
      		{
				A[i * (NQ * NP) + j * NP + k] = ((DATA_TYPE) i*j + k) / NP;
      		}
    	}
  	}	 
  
	for (i = 0; i < NP; i++)
  	{
    		for (j = 0; j < NP; j++)
    		{
      			C4[i * NP + j] = ((DATA_TYPE) i*j) / NP;
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

void cl_mem_init(DATA_TYPE *A, DATA_TYPE *C4, DATA_TYPE* sum)
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, NR * NQ * NP * sizeof(DATA_TYPE), NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	b_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, NP * NP * sizeof(DATA_TYPE), NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	c_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, NR * NQ * NP * sizeof(DATA_TYPE), NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, NR * NQ * NP * sizeof(DATA_TYPE), A, 0, NULL, NULL);
	if(errcode != CL_SUCCESS)printf("Error in writing buffers\n");
	errcode = clEnqueueWriteBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, NP * NP * sizeof(DATA_TYPE), C4, 0, NULL, NULL);
	if(errcode != CL_SUCCESS)printf("Error in writing buffers\n");
	errcode = clEnqueueWriteBuffer(clCommandQue, c_mem_obj, CL_TRUE, 0, NR * NQ * NP * sizeof(DATA_TYPE), sum, 0, NULL, NULL);
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
	clKernel1 = clCreateKernel(clProgram, "doitgen_kernel1", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");
	clKernel2 = clCreateKernel(clProgram, "doitgen_kernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel2\n");
	clFinish(clCommandQue);
}

void cl_launch_kernel1(int r)
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)NP) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)NQ) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&c_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 3, sizeof(int), (void *)&r);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clFinish(clCommandQue);
}

void cl_launch_kernel2(int r)
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)NP) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)NQ) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void *)&c_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 3, sizeof(int), (void *)&r);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
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
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(b_mem_obj);
	errcode = clReleaseMemObject(c_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}

void doitgen(DATA_TYPE *sum, DATA_TYPE *A, DATA_TYPE *C4)
{
	int r, q, p, s;

	for (r = 0; r < NR; r++)
	{
    	for (q = 0; q < NQ; q++)  
		{
			for (p = 0; p < NP; p++)  
			{
				sum[r * (NQ * NP) + q * NP + p] = (DATA_TYPE)0.0;
				for (s = 0; s < NP; s++)
				{
					sum[r * (NQ * NP) + q * NP + p] = sum[r * (NQ * NP) + q * NP + p] + A[r * (NQ * NP) + q * NP + s] * C4[s * NP + p];
				}
			}
      		for (p = 0; p < NP; p++)
       		{
				A[r * (NQ * NP) + q * NP + p] = sum[r * (NQ * NP) + q * NP + p];
			}
		}
	}
}

int main(void) {
	double t_start, t_end;
	int i;

	DATA_TYPE* A, *A_2;
	DATA_TYPE* C4, *C4_2;
	DATA_TYPE* sum, *sum_2;

	A = (DATA_TYPE*)malloc(NR * NQ * NP * sizeof(DATA_TYPE));
	C4 = (DATA_TYPE*)malloc(NP * NP * sizeof(DATA_TYPE));
	sum = (DATA_TYPE*)malloc(NR * NQ * NP * sizeof(DATA_TYPE));

	A_2 = (DATA_TYPE*)malloc(NR * NQ * NP * sizeof(DATA_TYPE));
	C4_2 = (DATA_TYPE*)malloc(NP * NP * sizeof(DATA_TYPE));
	sum_2 = (DATA_TYPE*)malloc(NR * NQ * NP * sizeof(DATA_TYPE));

	init_array(A, C4);
	init_array(A_2, C4_2);

	read_cl_file();
	cl_initialization();
	cl_mem_init(A, C4, sum);
	cl_load_prog();
	t_start = rtclock();

	int r;	

	for (r = 0; r < NR; r++)
	{
		cl_launch_kernel1(r);
		cl_launch_kernel2(r);
	}

	t_end = rtclock();
	errcode = clEnqueueReadBuffer(clCommandQue, c_mem_obj, CL_TRUE, 0, NR * NQ * NP * sizeof(DATA_TYPE), sum, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");
	
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	t_start = rtclock();
	doitgen(sum_2, A_2, C4_2);
	t_end = rtclock(); 
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);   
	compareResults(sum, sum_2);
	cl_clean_up();
    	return 0;
}

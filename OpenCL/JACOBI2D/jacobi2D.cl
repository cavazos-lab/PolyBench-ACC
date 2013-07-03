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

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

typedef float DATA_TYPE;
#define N 4096

__kernel void runJacobi2D_kernel1(__global DATA_TYPE* A, __global DATA_TYPE* B)
{
	int i = get_global_id(1);
	int j = get_global_id(0);

	if ((i > 1) && (i < (N-1)) && (j > 1) && (j < (N-1)))
	{
		B[i*N + j] = 0.2f * (A[i*N + j] + A[i*N + (j-1)] + A[i*N + (1 + j)] + A[(1 + i)*N + j] + A[(i-1)*N + j]);	
	}
}

__kernel void runJacobi2D_kernel2(__global DATA_TYPE* A, __global DATA_TYPE* B)
{
	int i = get_global_id(1);
	int j = get_global_id(0);
	
	if ((i > 1) && (i < (N-1)) && (j > 1) && (j < (N-1)))
	{
		A[i*N + j] = B[i*N + j];
	}
}

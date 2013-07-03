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

#ifndef N
# define N 4096
#endif

typedef float DATA_TYPE;



__kernel void lu_kernel1(__global DATA_TYPE *A, int k)
{
	int j = get_global_id(0) + (k + 1);
	
	if ((j < N))
	{
		A[k*N + j] = A[k*N + j] / A[k*N + k];
	}
}

__kernel void lu_kernel2(__global DATA_TYPE *A, int k)
{
	int j = get_global_id(0) + (k + 1);
	int i = get_global_id(1) + (k + 1);
	
	if ((i < N) && (j < N))
	{
		A[i*N + j] = A[i*N + j] - A[i*N + k] * A[k*N + j];
	}
}

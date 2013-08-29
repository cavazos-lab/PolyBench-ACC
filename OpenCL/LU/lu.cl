/**
 * lu.cl: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#ifndef N
# define N 2048
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

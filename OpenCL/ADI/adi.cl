/**
 * adi.cl: This file is part of the PolyBench/GPU 1.0 test suite.
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

typedef float DATA_TYPE;

#ifndef N
# define N 1024
#endif

__kernel void adi_kernel1(__global DATA_TYPE* A, __global DATA_TYPE* B, __global DATA_TYPE* X)
{
	int i1 = get_global_id(0);
	int i2;	

	if ((i1 < N))
	{
		for (i2 = 1; i2 < N; i2++)
		{
			X[i1*N + i2] = X[i1*N + i2] - X[i1*N + (i2-1)] * A[i1*N + i2] / B[i1*N + (i2-1)];
			B[i1*N + i2] = B[i1*N + i2] - A[i1*N + i2] * A[i1*N + i2] / B[i1*N + (i2-1)];
		}
	}
}

__kernel void adi_kernel2(__global DATA_TYPE* A, __global DATA_TYPE* B, __global DATA_TYPE* X)
{
	int i1 = get_global_id(0);
	
	if ((i1 < N))
	{
		X[i1*N + (N-1)] = X[i1*N + (N-1)] / B[i1*N + (N-1)];
	}
}
	
__kernel void adi_kernel3(__global DATA_TYPE* A, __global DATA_TYPE* B, __global DATA_TYPE* X)
{
	int i1 = get_global_id(0);
	int i2;	

	if ((i1 < N))
	{
		for (i2 = 0; i2 < N-2; i2++)
		{
			X[i1*N + (N-i2-2)] = (X[i1*N + (N-2-i2)] - X[i1*N + (N-2-i2-1)] * A[i1*N + (N-i2-3)]) / B[i1*N + (N-3-i2)];
		}
	}
}



__kernel void adi_kernel4(__global DATA_TYPE* A, __global DATA_TYPE* B, __global DATA_TYPE* X, int i1)
{
	int i2 = get_global_id(0);
	
	if ((i2 < N))
	{
		X[i1*N + i2] = X[i1*N + i2] - X[(i1-1)*N + i2] * A[i1*N + i2] / B[(i1-1)*N + i2];
		B[i1*N + i2] = B[i1*N + i2] - A[i1*N + i2] * A[i1*N + i2] / B[(i1-1)*N + i2];
	}
}

__kernel void adi_kernel5(__global DATA_TYPE* A, __global DATA_TYPE* B, __global DATA_TYPE* X)
{
	int i2 = get_global_id(0);
	
	if ((i2 < N))
	{
		X[(N-1)*N + i2] = X[(N-1)*N + i2] / B[(N-1)*N + i2];
	}
}

__kernel void adi_kernel6(__global DATA_TYPE* A, __global DATA_TYPE* B, __global DATA_TYPE* X, int i1)
{
	int i2 = get_global_id(0);
	
	if ((i2 < N))
	{
	     X[(N-2-i1)*N + i2] = (X[(N-2-i1)*N + i2] - X[(N-i1-3)*N + i2] * A[(N-3-i1)*N + i2]) / B[(N-2-i1)*N + i2];
	}
}


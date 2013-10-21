/**
 * 2mm.cl: This file is part of the PolyBench/GPU 1.0 test suite.
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

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

__kernel void mm2_kernel1(__global DATA_TYPE *tmp, __global DATA_TYPE *A, __global DATA_TYPE *B, int ni, int nj, int nk, int nl, DATA_TYPE alpha, DATA_TYPE beta) 
{    
	int j = get_global_id(0);
	int i = get_global_id(1);
	
	if ((i < ni) && (j < nj))
	{ 
		tmp[i * nj + j] = 0;
		int k;
		for (k = 0; k < nk; k++)
		{
			tmp[i * nj + j] += alpha * A[i * nk + k] * B[k * nj + j];
		}
	}
}


__kernel void mm2_kernel2(__global DATA_TYPE *tmp, __global DATA_TYPE *C, __global DATA_TYPE *D, int ni, int nj, int nk, int nl, DATA_TYPE alpha, DATA_TYPE beta) 
{    
	int j = get_global_id(0);
	int i = get_global_id(1);
	
	if ((i < ni) && (j < nl))
	{ 
		D[i * nl + j] *= beta;
		int k;
		for (k = 0; k < nj; k++)
		{
			D[i * nl + j] += tmp[i * nj + k] * C[k * nl + j];
		}
	}
}

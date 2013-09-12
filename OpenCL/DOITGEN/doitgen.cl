/**
 * doitgen.cl: This file is part of the PolyBench/GPU 1.0 test suite.
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


__kernel void doitgen_kernel1(int nr, int nq, int np, __global DATA_TYPE *A, __global DATA_TYPE *C4, __global DATA_TYPE *sum, int r)
{
	int p = get_global_id(0);
	int q = get_global_id(1);

	if ((p < np) && (q < nq))
	{
		sum[r * (nq * np) + q * np + p] = (DATA_TYPE)0.0;
	
		for (int s = 0; s < np; s++)
		{
			sum[r * (nq * np) + q * np + p] = sum[r * (nq * np) + q * np + p] + A[r * (nq * np) + q * np + s] * C4[s * np + p];
		}
	}
}

__kernel void doitgen_kernel2(int nr, int nq, int np, __global DATA_TYPE *A, __global DATA_TYPE *C4, __global DATA_TYPE *sum, int r)
{
	int p = get_global_id(0);
	int q = get_global_id(1);

	if ((p < np) && (q < nq))
	{
		A[r * (nq * np) + q * np + p] = sum[r * (nq * np) + q * np + p];
	}
}

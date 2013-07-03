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


typedef double DATA_TYPE;
#ifndef NR
# define NR 512
#endif
#ifndef NQ
# define NQ 512
#endif
#ifndef NP
# define NP 512
#endif



__kernel void doitgen_kernel1(__global DATA_TYPE *A, __global DATA_TYPE *C4, __global DATA_TYPE *sum, int r)
{
	int p = get_global_id(0);
	int q = get_global_id(1);

	if ((p < NP) && (q < NQ))
	{

		sum[r * (NQ * NP) + q * NP + p] = (DATA_TYPE)0.0;
	
		int s;

		for (s = 0; s < NP; s++)
		{
			sum[r * (NQ * NP) + q * NP + p] = sum[r * (NQ * NP) + q * NP + p] + A[r * (NQ * NP) + q * NP + s] * C4[s * NP + p];
		}
	}
}

__kernel void doitgen_kernel2(__global DATA_TYPE *A, __global DATA_TYPE *C4, __global DATA_TYPE *sum, int r)
{
	int p = get_global_id(0);
	int q = get_global_id(1);

	if ((p < NP) && (q < NQ))
	{
		A[r * (NQ * NP) + q * NP + p] = sum[r * (NQ * NP) + q * NP + p];
	}
}

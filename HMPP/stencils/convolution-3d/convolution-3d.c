/* POLYBENCH/GPU-HMPP
 *
 * This file is a part of the Polybench/GPU-HMPP suite
 *
 * Contact:
 * William Killian <killian@udel.edu>
 * 
 * Copyright 2013, The University of Delaware
 */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4096x4096. */
#include "convolution-3d.h"


/* Array initialization. */
static
void init_array (int ni, int nj, int nk,
		 DATA_TYPE POLYBENCH_3D(A,NI,NJ,NK,ni,nj,nk))
{
  int i, j, k;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      for (k = 0; j < nk; k++)
	{
	  A[i][j][k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
	}
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj, int nk,
		 DATA_TYPE POLYBENCH_3D(B,NI,NJ,NK,ni,nj,nk))

{
  int i, j, k;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      for (k = 0; j < nk; k++) {
	fprintf(stderr, DATA_PRINTF_MODIFIER, B[i][j][k]);
	if (((i * NJ + j) * NK + k) % 20 == 0) fprintf(stderr, "\n");
      }
  fprintf(stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
#pragma hmpp conv3d codelet, &
#pragma hmpp & args[ni;nj;nk].transfer=atcall, &
#pragma hmpp & args[A;B].transfer=manual, &
#pragma hmpp & target=CUDA:OPENCL
static
void kernel_conv2d(int ni,
		   int nj,
		   int nk,
		   DATA_TYPE POLYBENCH_3D(A,NI,NJ,NK,ni,nj,nk),
		   DATA_TYPE POLYBENCH_3D(B,NI,NJ,NK,ni,nj,nk))
{
  int i, j, k;
  for (i = 1; i < _PB_NI - 1; ++i)
    for (j = 1; j < _PB_NJ - 1; ++j)
      for (k = 1; k < _PB_NK - 1; ++k)
	{
	  B[i][j][k]
	    =  2 * A[i-1][j-1][k-1]  +  4 * A[i+1][j-1][k-1]
	    +  5 * A[i-1][j-1][k-1]  +  7 * A[i+1][j-1][k-1]
	    + -8 * A[i-1][j-1][k-1]  + 10 * A[i+1][j-1][k-1]
	    + -3 * A[ i ][j-1][ k ]
	    +  6 * A[ i ][ j ][ k ]
	    + -9 * A[ i ][j+1][ k ]
	    +  2 * A[i-1][j-1][k+1]  +  4 * A[i+1][j-1][k+1]
	    +  5 * A[i-1][ j ][k+1]  +  7 * A[i+1][ j ][k+1]
	    + -8 * A[i-1][j+1][k+1]  + 10 * A[i+1][j+1][k+1];
	}
}


int main(int argc, char** argv)
{
  #pragma hmpp conv3d acquire

  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;

  /* Variable declaration/allocation. */
  POLYBENCH_3D_ARRAY_DECL(A, DATA_TYPE, NI, NJ, NK, ni, nj, nk);
  POLYBENCH_3D_ARRAY_DECL(B, DATA_TYPE, NI, NJ, NK, ni, nj, nk);

  #pragma hmpp conv3d allocate, &
  #pragma hmpp & args[A].size={ni,nj,nk}, args[A].hostdata="A", &
  #pragma hmpp & args[B].size={ni,nj,nk}, args[B].hostdata="B"

  /* Initialize array(s). */
  init_array (ni, nj, nk, POLYBENCH_ARRAY(A));
  
  #pragma hmpp conv3d advancedload, args[A]
  
  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  #pragma hmpp conv3d callsite
  kernel_conv2d (ni, nj, nk, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;
  
  #pragma hmpp conv3d delegatedstore, args[B]
  
  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nj, nk, POLYBENCH_ARRAY(B)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  #pragma hmpp conv3d release
  
  return 0;
}

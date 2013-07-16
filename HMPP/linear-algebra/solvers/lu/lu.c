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
/* Default data type is double, default size is 1024. */
#include "lu.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      A[i][j] = ((DATA_TYPE) (i+1)*(j+1)) / n;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      fprintf (stderr, DATA_PRINTF_MODIFIER, A[i][j]);
      if ((i * n + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
#pragma hmpp lu codelet, &
#pragma hmpp & args[n].transfer=atcall, &
#pragma hmpp & args[A].transfer=manual, &
#pragma hmpp & target=CUDA:OPENCL
static
void kernel_lu(int n,
	       DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j, k;
  for (k = 0; k < _PB_N; k++)
    {
      for (j = k + 1; j < _PB_N; j++)
	A[k][j] = A[k][j] / A[k][k];
      for(i = k + 1; i < _PB_N; i++)
	for (j = k + 1; j < _PB_N; j++)
	  A[i][j] = A[i][j] - A[i][k] * A[k][j];
    }
}


int main(int argc, char** argv)
{
  #pragma hmpp lu acquire
  
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);

  #pragma hmpp lu allocate, &
  #pragma hmpp & args[A].size={n,n}, args[A].hostdata="A"

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A));

  #pragma hmpp lu advancedload, args[A]
  
  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  #pragma hmpp lu callsite
  kernel_lu (n, POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;
  
  #pragma hmpp lu delegatedstore, args[A]
  
  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  #pragma hmpp lu release
  
  return 0;
}

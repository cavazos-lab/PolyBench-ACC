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
/* Default data type is double, default size is 4000. */
#include "trisolv.h"


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		DATA_TYPE POLYBENCH_1D(x,N,n),
		DATA_TYPE POLYBENCH_1D(c,N,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      c[i] = x[i] = ((DATA_TYPE) i) / n;
      for (j = 0; j < n; j++)
	A[i][j] = ((DATA_TYPE) i*j) / n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x,N,n))

{
  int i;

  for (i = 0; i < n; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, x[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
#pragma hmpp trisolve codelet, &
#pragma hmpp & args[n].transfer=atcall, &
#pragma hmpp & args[A;x;c].transfer=manual, &
#pragma hmpp & target=CUDA:OPENCL
static
void kernel_trisolv(int n,
		    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		    DATA_TYPE POLYBENCH_1D(x,N,n),
		    DATA_TYPE POLYBENCH_1D(c,N,n))
{
  int i, j;
  for (i = 0; i < _PB_N; i++)
    {
      x[i] = c[i];
      for (j = 0; j <= i - 1; j++)
	x[i] = x[i] - A[i][j] * x[j];
      x[i] = x[i] / A[i][i];
    }
}


int main(int argc, char** argv)
{
  #pragma hmpp trisolve acquire

  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(c, DATA_TYPE, N, n);

  #pragma hmpp trisolve allocate, &
  #pragma hmpp & args[A].size={n,n}, args[A].hostdata="A", &
  #pragma hmpp & args[x].size={n}, args[x].hostdata="x", &
  #pragma hmpp & args[c].size={n}, args[c].hostdata="c"
  
  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(c));

  #pragma hmpp trisolve advancedload, args[A;x;c]

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  #pragma hmpp trisolve callsite
  kernel_trisolv (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(c));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  #pragma hmpp trisolve delegatedstore, args[x]

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(c);

  #pragma hmpp trisolve release

  return 0;
}

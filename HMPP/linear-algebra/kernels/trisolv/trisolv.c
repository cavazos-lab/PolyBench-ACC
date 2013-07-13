/**
 * trisolv.c: This file is part of the PolyBench/C 3.2 test suite.
 *
 *
 * Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://polybench.sourceforge.net
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
static
void kernel_trisolv(int n,
		    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		    DATA_TYPE POLYBENCH_1D(x,N,n),
		    DATA_TYPE POLYBENCH_1D(c,N,n))
{
  int i, j;
  
  #pragma scop
  #pragma hmpp trisolve acquire
  // timing start
  // data transfer start
  #pragma hmpp trisolve allocate, &
  #pragma hmpp & args[n], &
  #pragma hmpp & args[x;c].size={n}, &
  #pragma hmpp & args[A].size={n,n}
  
  #pragma hmpp trisolve advancedload, &
  #pragma hmpp & args[n], &
  #pragma hmpp & args[A;x;c]
  // data transfer stop
  // kernel start
  #pragma hmpp trisolve region, &
  #pragma hmpp & args[*].transfer=manual, &
  #pragma hmpp & target=CUDA, &
  #pragma hmpp & asynchronous
  {
    for (i = 0; i < _PB_N; i++)
      {
	x[i] = c[i];
	for (j = 0; j <= i - 1; j++)
	  x[i] = x[i] - A[i][j] * x[j];
	x[i] = x[i] / A[i][i];
      }
  }
  #pragma hmpp trisolve synchronize
  // kernel stop
  // data transfer start
  #pragma hmpp trisolve delegatedstore, args[x]
  // data transfer stop
  // timing stop
  #pragma hmpp trisolve release
  #pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(c, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(c));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_trisolv (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(c));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(c);

  return 0;
}

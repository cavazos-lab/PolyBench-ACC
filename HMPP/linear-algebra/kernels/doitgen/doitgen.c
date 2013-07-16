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
#include "doitgen.h"


/* Array initialization. */
static
void init_array(int nr, int nq, int np,
		DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np))
{
  int i, j, k;

  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++)
	A[i][j][k] = ((DATA_TYPE) i*j + k) / np;
  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++)
      C4[i][j] = ((DATA_TYPE) i*j) / np;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nr, int nq, int np,
		 DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np))
{
  int i, j, k;

  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, A[i][j][k]);
	if (i % 20 == 0) fprintf (stderr, "\n");
      }
  fprintf (stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
#pragma hmpp doitgen codelet, &
#pragma hmpp & args[nr;nq;np].transfer=atcall, &
#pragma hmpp & args[A;C4;sum].transfer=manual, &
#pragma hmpp & target=CUDA:OPENCL
static
void kernel_doitgen(int nr, int nq, int np,
		    DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		    DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
		    DATA_TYPE POLYBENCH_3D(sum,NR,NQ,NP,nr,nq,np))
{
  int r, q, p, s;
  
  for (r = 0; r < _PB_NR; r++)
    for (q = 0; q < _PB_NQ; q++) 
      {
	for (p = 0; p < _PB_NP; p++)
	  {
	    sum[r][q][p] = 0;
	    for (s = 0; s < _PB_NP; s++)
	      sum[r][q][p] = sum[r][q][p] + A[r][q][s] * C4[s][p];
	  }
	for (p = 0; p < _PB_NR; p++)
	  A[r][q][p] = sum[r][q][p];
      }
}

int main(int argc, char** argv)
{
  #pragma hmpp doitgen acquire

  /* Retrieve problem size. */
  int nr = NR;
  int nq = NQ;
  int np = NP;

  /* Variable declaration/allocation. */
  POLYBENCH_3D_ARRAY_DECL(A,DATA_TYPE,NR,NQ,NP,nr,nq,np);
  POLYBENCH_3D_ARRAY_DECL(sum,DATA_TYPE,NR,NQ,NP,nr,nq,np);
  POLYBENCH_2D_ARRAY_DECL(C4,DATA_TYPE,NP,NP,np,np);
  
  #pragma hmpp doitgen allocate, &
  #pragma hmpp & args[A].size={nr,nq,np}, args[A].hostdata="A", &
  #pragma hmpp & args[sum].size={nr,nq,np}, args[sum].hostdata="sum", &
  #pragma hmpp & args[C4].size={np,np}, args[C4].hostdata="C4"
  
  /* Initialize array(s). */
  init_array (nr, nq, np,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(C4));

  #pragma hmpp doitgen advancedload, args[A;C4]

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  #pragma hmpp doitgen callsite
  kernel_doitgen (nr, nq, np,
		  POLYBENCH_ARRAY(A),
		  POLYBENCH_ARRAY(C4),
		  POLYBENCH_ARRAY(sum));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  #pragma hmpp doitgen delegatedstore, args[A]

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nr, nq, np,  POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(sum);
  POLYBENCH_FREE_ARRAY(C4);

  #pragma hmpp doitgen release

  return 0;
}

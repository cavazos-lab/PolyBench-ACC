/**
 * doitgen.cuh: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#ifndef DOITGEN_CUH
# define DOITGEN_CUH

/* Default to STANDARD_DATASET. */
# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#  define STANDARD_DATASET
# endif

/* Do not define anything if the user manually defines the size. */
# if !defined(NQ) && !defined(NR) && !defined(NP)
/* Define the possible dataset sizes. */
#  ifdef MINI_DATASET
#   define NQ 32
#   define NR 32
#   define NP 32
#  endif

#  ifdef SMALL_DATASET
#   define NQ 64
#   define NR 64
#   define NP 64
#  endif

#  ifdef STANDARD_DATASET /* Default if unspecified. */
#   define NQ 128
#   define NR 128
#   define NP 128
#  endif

#  ifdef LARGE_DATASET
#   define NQ 256
#   define NR 256
#   define NP 256
#  endif

#  ifdef EXTRALARGE_DATASET
#   define NQ 512
#   define NR 512
#   define NP 512
#  endif
# endif /* !N */

# define _PB_NQ POLYBENCH_LOOP_BOUND(NQ,nq)
# define _PB_NR POLYBENCH_LOOP_BOUND(NR,nr)
# define _PB_NP POLYBENCH_LOOP_BOUND(NP,np)

# ifndef DATA_TYPE
#  define DATA_TYPE float
#  define DATA_PRINTF_MODIFIER "%0.2lf "
# endif

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

#endif /* !DOITGEN */

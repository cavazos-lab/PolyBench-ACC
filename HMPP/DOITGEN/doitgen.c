#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>

#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size. */
#define NR 128
#define NQ 128
#define NP 128

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



#pragma hmpp conv codelet, target=OpenCL, args[a, p_sum].io=inout
void loop(DATA_TYPE a[NR][NQ][NP], DATA_TYPE b[NR][NP], DATA_TYPE p_sum[NR][NQ][NP])
{
   	int r, q, p, s;
   	int nr = 128;
   	int np = 128;
   	int nq = 128;

	# pragma hmppcg grid blocksize 32 X 4
   	for (r = 0; r < nr; r++)
	{
    		for (q = 0; q < nq; q++)  
		{
      			for (p = 0; p < np; p++)  
			{
        			p_sum[r][q][p] = 0;
		
        			for (s = 0; s < np; s++)
				{
          				p_sum[r][q][p] = p_sum[r][q][p] + a[r][q][s] * b[s][p];
				}
      			}
	  
      			for (p = 0; p < np; p++)
			{
        			a[r][q][p] = p_sum[r][q][p];
			}
    		}
	}
}


void compareResults(DATA_TYPE a[NR][NQ][NP], DATA_TYPE b[NR][NQ][NP], DATA_TYPE c[NR][NQ][NP], DATA_TYPE d[NR][NQ][NP])
{
	int i, j, k, fail;
	fail = 0;
	
	// Compare a and b
	for (i=0; i<NR; i++) 
	{
		for (j=0; j<NQ; j++) 
		{
			for(k=0; k<NP; k++)
			{
				if (percentDiff(a[i][j][k], b[i][j][k]) > PERCENT_DIFF_ERROR_THRESHOLD)
				{
					fail++;
				}
				if (percentDiff(c[i][j][k], d[i][j][k]) > PERCENT_DIFF_ERROR_THRESHOLD)
				{
					fail++;
				}
			}
		}
	}
	
	// Print results
	printf("Number of misses: %d\n", fail);
}


void init_array(DATA_TYPE A[NR][NQ][NP], DATA_TYPE Ai[NR][NQ][NP], DATA_TYPE C4[NP][NP])
{
  	int i, j, k;
  	float temp;

  	for (i = 0; i < NR; i++)
	{
    		for (j = 0; j < NQ; j++)
		{
      			for (k = 0; k < NP; k++)
			{
				A[i][j][k] = ((DATA_TYPE) i*j + k) / NP;
        			temp = A[i][j][k];
				Ai[i][j][k] = temp;
       		}
		}
	}

  	for (i = 0; i < NP; i++)
	{
    		for (j = 0; j < NP; j++)
		{
      			C4[i][j] = ((DATA_TYPE) i*j) / NP;
		}
	}
}



int main(int argc, char** argv)
{
    	double t_start, t_end;

	/* Array declaration */
	DATA_TYPE A[NR][NQ][NP];
	DATA_TYPE Ai[NR][NQ][NP];
	DATA_TYPE sum[NR][NQ][NP];
	DATA_TYPE C4[NP][NP];
	DATA_TYPE par_sum[NR][NQ][NP];
	
	#pragma hmpp conv allocate
	
	#pragma hmpp conv advancedload, args[a,b,p_sum]

  	/* Initialize array. */
  	init_array(A, Ai, C4);

  	t_start = rtclock();
	
 	#pragma hmpp conv callsite, args[a,b,p_sum].advancedload=true, asynchronous
  	loop(Ai, C4, par_sum);
	
	#pragma hmpp conv synchronize

  	t_end = rtclock();
  	fprintf(stderr, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	
	#pragma hmpp conv delegatedstore, args[a,p_sum]
	#pragma hmpp conv release

  	t_start = rtclock();

  	loop(A, C4, sum);  

  	t_end = rtclock();
  	fprintf(stderr, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  	compareResults(par_sum, sum, A, Ai);

  	return 0;
}


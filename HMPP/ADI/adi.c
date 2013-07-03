#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size */
#define TSTEPS 10
#define N 1024

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



#pragma hmpp conv codelet, target=OpenCL, args[X,B].io=inout
void adi(DATA_TYPE X[N][N], DATA_TYPE A[N][N], DATA_TYPE B[N][N])
{
	int t, i1, i2;
  	int n = N;
	int tsteps = 10;

  	for (t = 0; t < tsteps; t++)
    	{
		for (i1 = 0; i1 < n; i1++)
		{
			for (i2 = 1; i2 < n; i2++)
          		{
            			X[i1][i2] = X[i1][i2] - X[i1][i2-1] * A[i1][i2] / B[i1][i2-1];
            			B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][i2-1];
          		}
		}


     		for (i1 = 0; i1 < n; i1++)
		{
      			X[i1][n-1] = X[i1][n-1] / B[i1][n-1];
		}


      		for (i1 = 0; i1 < n; i1++)
		{
        		for (i2 = 0; i2 < n-2; i2++)
          			X[i1][n-i2-2] = (X[i1][n-2-i2] - X[i1][n-2-i2-1] * A[i1][n-i2-3]) / B[i1][n-3-i2];
		}


      		for (i1 = 1; i1 < n; i1++)
		{
			for (i2 = 0; i2 < n; i2++) 
			{
          			X[i1][i2] = X[i1][i2] - X[i1-1][i2] * A[i1][i2] / B[i1-1][i2];
          			B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1-1][i2];
        		}  
		}

      		for (i2 = 0; i2 < n; i2++)
		{
        		X[n-1][i2] = X[n-1][i2] / B[n-1][i2];
		}

      		for (i1 = 0; i1 < n-2; i1++)
		{
        		for (i2 = 0; i2 < n; i2++)
			{
          			X[n-2-i1][i2] = (X[n-2-i1][i2] - X[n-i1-3][i2] * A[n-3-i1][i2]) / B[n-2-i1][i2];
			}
		}
    	}
}


void init_array(DATA_TYPE X[N][N], DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE X2[N][N], DATA_TYPE B2[N][N])
{ 
  	int i, j;

	for (i = 0; i < N; i++)
	{
    		for (j = 0; j < N; j++)
      		{ 
        		X[i][j] = ((DATA_TYPE) i*j + 1) / N;
        		A[i][j] = ((DATA_TYPE) i*j + 2) / N;
        		B[i][j] = ((DATA_TYPE) i*j + 3) / N;
			X2[i][j] = ((DATA_TYPE) i*j + 1) / N;
			B2[i][j] = ((DATA_TYPE) i*j + 3) / N;
		}
	}
}


void compareResults(DATA_TYPE a[N][N], DATA_TYPE b[N][N], DATA_TYPE c[N][N], DATA_TYPE d[N][N])
{
       int i, j, fail;
       fail = 0;   

       // Compare a and c
       for (i=0; i<N; i++) 
	{
		for (j=0; j<N; j++)
		{
			if (percentDiff(a[i][j], c[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
                   	}
		}
	}

	// Compare b and d
       for (i=0; i<N; i++) 
	{
		for (j=0; j<N; j++)
		{
			if (percentDiff(b[i][j], d[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}

	// Print results
	printf("Number of misses: %d\n", fail);
}


int main(int argc, char** argv)
{
	double t_start, t_end;
	
	/* Array declaration. */
	DATA_TYPE X[N][N];
	DATA_TYPE A[N][N];
	DATA_TYPE B[N][N];
	DATA_TYPE X2[N][N];
	DATA_TYPE B2[N][N];

	/* Initialize array. */
	init_array(X, A, B, X2, B2);

       #pragma hmpp conv allocate

       #pragma hmpp conv advancedload, args[X;B]
       
	// Run GPU code

       t_start = rtclock();

       #pragma hmpp conv callsite, args[X;B].advancedload=true, asynchronous
       adi(X, A, B);

       #pragma hmpp conv synchronize

   	t_end = rtclock();
   	fprintf(stderr, "GPU Runtime: %0.6lfs\n", t_end - t_start);

       #pragma hmpp conv delegatedstore, args[X;B]
       #pragma hmpp conv release

       /*IF_TIME(*/t_start = rtclock();//);

       adi(X2, A, B2);

    	t_end = rtclock();
    	fprintf(stderr, "CPU Runtime: %0.6lfs\n", t_end - t_start);

       compareResults(X, B, X2, B2);

	return 0;
}


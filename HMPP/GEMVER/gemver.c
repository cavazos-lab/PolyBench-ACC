/*********************************************************************************/
//
// Polybench kernels implementation using HMPP for execution on the GPU
//
// Computer & Information Science, University of Delaware
// Author(s):   Sudhee Ayalasomayajula (sudhee1@gmail.com)
//              John Cavazos (cavazos@cis.udel.edu)
//				Scott Grauer Gray(sgrauerg@gmail.com)
//              Robert Searles (rsearles35@aol.com)   
//              Lifan Xu (xulifan@udel.edu)
//
// Contact(s): Scott Grauer Gray(sgrauerg@gmail.com)
// Reference(s):
//
/*********************************************************************************/

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

/* Problem size */
#define N 4096

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



#pragma hmpp conv codelet, target=OpenCL, args[w].io=inout
void loop(DATA_TYPE A[N][N], DATA_TYPE x[N], DATA_TYPE u1[N], DATA_TYPE u2[N], DATA_TYPE v2[N], DATA_TYPE v1[N],
			DATA_TYPE w[N], DATA_TYPE y[N], DATA_TYPE z[N])
{
	int i, j, k;

	DATA_TYPE alpha = 43532;
	DATA_TYPE beta = 12313;
	
	#pragma hmppcg grid blocksize 32 X 8
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
		}
	}
	
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			x[i] = x[i] + beta * A[j][i] * y[j];
		}
	}

	for (i = 0; i < N; i++)
	{
		x[i] = x[i] + z[i];
	}

	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			w[i] = w[i] +  alpha * A[i][j] * x[j];
		}
	}
}


void init(DATA_TYPE A[N][N], DATA_TYPE x[N], DATA_TYPE y[N], DATA_TYPE z[N], DATA_TYPE w[N], DATA_TYPE wi[N], DATA_TYPE v1[N],
			DATA_TYPE v2[N], DATA_TYPE u1[N], DATA_TYPE u2[N])
{
 	int i, j;

  	for (i = 0; i < N; i++)
    {
    	u1[i] = i;
    	u2[i] = (i+1)/N/2.0;
    	v1[i] = (i+1)/N/4.0;
    	v2[i] = (i+1)/N/6.0;
    	y[i] = (i+1)/N/8.0;
    	z[i] = (i+1)/N/9.0;
    	x[i] = 0.0;
    	w[i] = 0.0;
		wi[i] = w[i];

    	for (j = 0; j < N; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / N;
		}
    }
}

void compareResults(DATA_TYPE w1[N], DATA_TYPE w2[N])
{
	int i, fail;
	fail = 0;
	
	for (i=0; i < N; i++) 
	{
		if (percentDiff(w1[i], w2[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
		{
			fail++;
		}
	}
		
	// Print results
	printf("Number of misses: %d\n", fail);
}


int main(int argc, char** argv)
{
	double t_start, t_end;

	/* Array declaration.  */
	DATA_TYPE A[N][N];
	DATA_TYPE x[N];
	DATA_TYPE u1[N];
	DATA_TYPE u2[N];
	DATA_TYPE v2[N];
	DATA_TYPE v1[N];
	DATA_TYPE w[N];
	DATA_TYPE wi[N];
	DATA_TYPE y[N];
	DATA_TYPE z[N];

	/* Initialize array. */
	init(A, x, u1, u2, v2, v1, w, wi, y, z);
    
	#pragma hmpp conv allocate
	#pragma hmpp conv advancedload, args[A;x;u1;u2;v2;v1;w;y;z]

	t_start = rtclock();

	#pragma hmpp conv callsite, args[A;x;u1;u2;v2;v1;w;y;z].advancedload=true, asynchronous
	loop(A, x, u1, u2, v2, v1, wi, y, z);
	#pragma hmpp conv synchronize

	t_end = rtclock();//);
	fprintf(stderr, "GPU Runtime: %0.6lfs\n", t_end - t_start);
    
	#pragma hmpp conv delegatedstore, args[w]
	#pragma hmpp conv release
	
	t_start = rtclock();

	loop(A, x, u1, u2, v2, v1, w, y, z);

	t_end = rtclock();
	fprintf(stderr, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	compareResults(w, wi);

	return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <omp.h>
#include <unistd.h>
#include <sys/time.h>

#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size */
#define N 2048

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void init_array(DATA_TYPE A[N][N], DATA_TYPE B[N][N])
{
	int i, j;

	for (i=0; i< N; i++) 
	{
		for (j=0; j< N; j++)
		{
			A[i][j] = (DATA_TYPE)(i+j+1.0);
			B[i][j] = (DATA_TYPE)(i+j+1.0);
		}
	}
}


#pragma hmpp lu codelet, target=OpenCL, args[A].io=inout
void runLu(DATA_TYPE A[N][N])
{	
	int i, j, k;
	
	# pragma hmppcg grid blocksize 32 X 4	
	for (k=0; k<N; k++) 
	{
		
        	for (j=k+1; j<N; j++)   
		{
            		A[k][j] = A[k][j]/A[k][k];

            		for(i=k+1; i<N; i++)    
			{
                		A[i][j] = A[i][j] - A[i][k]*A[k][j];
            		}
        	}
    	}
}


void compareResults(DATA_TYPE A[N][N], DATA_TYPE B[N][N])
{
	int i,j,fail;
	fail = 0;
	
	for (i=0; i<N; i++) 
	{
		for (j=0; j<N; j++) 
		{
			if (percentDiff(A[i][j], B[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}
		}
	}
	
	printf("Number of misses: %d\n", fail);
}


int main()
{
    	double t_start, t_end;

	DATA_TYPE A[N][N];
	DATA_TYPE B[N][N];

	init_array(A, B);

	#pragma hmpp lu allocate

	#pragma hmpp lu advancedload, args[A]

	t_start = rtclock();

	#pragma hmpp lu callsite, args[A].advancedload=true, asynchronous
	runLu(A);

	#pragma hmpp lu synchronize

	t_end = rtclock();
	fprintf(stderr, "GPU Runtime: %0.6lf\n", t_end - t_start);
	
	#pragma hmpp lu delegatedstore, args[A]

	#pragma hmpp lu release

	t_start = rtclock();
	
	runLu(B);
	
	t_end = rtclock();
	fprintf(stderr, "CPU Runtime: %0.6lf\n", t_end - t_start);
	
	compareResults(A, B);

 	return 0;
}

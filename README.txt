README

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
* PolyBench/GPU 1.0:  PolyBench Benchmarks on the GPU using CUDA, OpenCL, HMPP, and OpenACC.  *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

Copyright (c) 2012, 2013 University of Delaware
Contact:  Scott Grauer-Gray <sgrauerg@gmail.com>
		  William Killian <killian@udel.edu>
		  John Cavazos <cavazos@udel.edu>

This benchmark suite is partially derived from the PolyBench benchmark suite developed by Louis-Noel
Pouchet <pouchet@cse.ohio-state.edu> and available at http://www.cse.ohio-state.edu/~pouchet/software/polybench/
		  
If using this work, please cite the following paper: 
Scott Grauer-Gray, Lifan Xu, Robert Searles, Sudhee Ayalasomayajula, and John Cavazos.  
Auto-tuning a High-Level Language Targeted to GPU Codes. 
Proceedings of Innovative Parallel Computing 
(InPar '12), 2012.

Paper available at http://www.eecis.udel.edu/~grauerg/


-----------------------
* Available benchmarks:
-----------------------

Convolution:
2DCONV
3DCONV

Linear Algebra:
2MM
3MM
ATAX
BICG
DOITGEN
GEMM
GESUMMV
GRAMSCHMIDT
LU
MVT
SYR2K
SYRK

Datamining:
CORRELATION
COVARIANCE

Stencils:
ADI
FDTD-2D
JACOBI-1D
JACOBI-2D

The CUDA, OpenCL, HMPP, and OpenACC codes are based on PolyBench 3.2.


--------------------------------------------------------------------------------------------
* Instructions - to compile/run CUDA, OpenCL, and HMPP (OpenACC described separately below):
--------------------------------------------------------------------------------------------

CUDA: 
1. Set up PATH and LD_LIBRARY_PATH environment variables to point to CUDA installation 
2. Run "make" in target folder(s) with codes to generate executable(s)
3. Run the generated .exe file(s).

OpenCL:
1. Set up PATH and LD_LIBRARY_PATH environment variables to point to OpenCL installation
2. Set location of SDK in common.mk file in OpenCL folder
3. Run "make" in target folder(s) to generate executable(s)
4. Run the generated .exe file(s).

HMPP:
1. Change to bash shell if not currently in it.
2. Set up PATH and LD_LIBRARY_PATH environment variables to point to CUDA/OpenCL installation 
3. Set up HMPP environment variables with source hmpp-env.sh command in {HMPP_INSTALLATION}/bin folder 
4. Run "make exe" in target folder(s) with codes to generate executable(s)
5. Run generated .exe file(s).  If there's an error when running the .exe file(s), try running them with the "make" or "make run"
command in the folder. 


------------------
* Modifying Codes:
------------------

Parameters such as the input sizes, data type, and threshold for GPU-CPU output comparison can be modified using constants
within the codes.  After modifying, run "make clean" then "make" on relevant code for modifications to take effect in resulting executable.

NOTES ABOUT PARAMETERS:

DATA_TYPE:
By default, the DATA_TYPE used in these codes are floats; that can be changed to doubles by changing the DATA_TYPE typedef to "double"

PERCENT_DIFF_ERROR_THRESHOLD:
The PERCENT_DIFF_ERROR_THRESHOLD refers to the percent difference (0.0-100.0) that the GPU and CPU results are allowed to differ and still be considered "matching";
this parameter can be adjusted for each code in the input code file.

=======================================================================================================================================================================================
OPENACC INFO: 

** To compile OpenACC version using HMPP Workbench / CAPS Compiler:
-------------------------------------------------------------------

* Targeting CUDA:

$> hmpp --codelet-required --openacc-target=CUDA gcc -O2 -I./utilities -I./linear-algebra/kernels/gemm/gemm utilities/polybench.c linear-algebra/kernels/gemm/gemm.c -o gemm_acc
   
   OR
   
$> capsmc --codelet-required --openacc-target=CUDA gcc -O2 -I./utilities -I./linear-algebra/kernels/gemm/gemm utilities/polybench.c linear-algebra/kernels/gemm/gemm.c -o gemm_acc

* Targeting OpenCL:

$> hmpp --codelet-required --openacc-target=OPENCL gcc -O2 -I./utilities -I./linear-algebra/kernels/gemm/gemm utilities/polybench.c linear-algebra/kernels/gemm/gemm.c -o gemm_acc
   
   OR
   
$> capsmc --codelet-required --openacc-target=OPENCL gcc -O2 -I./utilities -I./linear-algebra/kernels/gemm/gemm utilities/polybench.c linear-algebra/kernels/gemm/gemm.c -o gemm_acc

** To generate the reference output of a benchmark:
---------------------------------------------------

* Pass the -DPOLYBENCH_DUMP_ARRAYS argument to the host compiler (i.e. gcc or icc) when compiling

$> ./gemm_acc 2>gemm_ref.out

------------------------------------
* Some available options (OpenACC):
------------------------------------

They are all passed as macro definitions during compilation time (e.g,
-Dname_of_the_option).

- POLYBENCH_TIME: output execution time (gettimeofday) [default: off]

- POLYBENCH_NO_FLUSH_CACHE: don't flush the cache before calling the
  timer [default: flush the cache]

- POLYBENCH_LINUX_FIFO_SCHEDULER: use FIFO real-time scheduler for the
  kernel execution, the program must be run as root, under linux only,
  and compiled with -lc [default: off]

- POLYBENCH_CACHE_SIZE_KB: cache size to flush, in kB [default: 33MB]

- POLYBENCH_STACK_ARRAYS: use stack allocation instead of malloc [default: off]

- POLYBENCH_DUMP_ARRAYS: dump all live-out arrays on stderr [default: off]

- POLYBENCH_CYCLE_ACCURATE_TIMER: Use Time Stamp Counter to monitor
  the execution time of the kernel [default: off]

- POLYBENCH_PAPI: turn on papi timing (see below).

- MINI_DATASET, SMALL_DATASET, STANDARD_DATASET, LARGE_DATASET,
  EXTRALARGE_DATASET: set the dataset size to be used
  [default: STANDARD_DATASET]

- POLYBENCH_USE_C99_PROTO: Use standard C99 prototype for the functions.

- POLYBENCH_USE_SCALAR_LB: Use scalar loop bounds instead of parametric ones.



--------------------------
* PAPI support (OpenACC):
--------------------------

** To compile a benchmark with PAPI support:
--------------------------------------------

$> gcc -O3 -I utilities -I linear-algebra/kernels/atax utilities/polybench.c linear-algebra/kernels/atax/atax.c -DPOLYBENCH_PAPI -lpapi -o atax_papi


** To specify which counter(s) to monitor:
------------------------------------------

Edit utilities/papi_counters.list, and add 1 line per event to
monitor. Each line (including the last one) must finish with a ',' and
both native and standard events are supported.

The whole kernel is run one time per counter (no multiplexing) and
there is no sampling being used for the counter value.



-----------------------------------------
* Accurate performance timing (OpenACC):
-----------------------------------------

With kernels that have an execution time in the orders of a few tens
of milliseconds, it is critical to validate any performance number by
repeating several times the experiment. A companion script is
available to perform reasonable performance measurement of a PolyBench.

$> gcc -O3 -I utilities -I linear-algebra/kernels/atax utilities/polybench.c linear-algebra/kernels/atax/atax.c -DPOLYBENCH_TIME -o atax_time
$> ./utilities/time_benchmark.sh ./atax_time

This script will run five times the benchmark (that must be a
PolyBench compiled with -DPOLYBENCH_TIME), eliminate the two extremal
times, and check that the deviation of the three remaining does not
exceed a given threshold, set to 5%.

It is also possible to use POLYBENCH_CYCLE_ACCURATE_TIMER to use the
Time Stamp Counter instead of gettimeofday() to monitor the number of
elapsed cycles.




---------------------------------------------------
* Generating macro-free benchmark suite (OpenACC):
---------------------------------------------------

(from the root of the archive:)
$> PARGS="-I utilities -DPOLYBENCH_TIME";
$> for i in `cat utilities/benchmark_list`; do create_cpped_version.sh $i "$PARGS"; done

This create for each benchmark file 'xxx.c' a new file
'xxx.preproc.c'. The PARGS variable in the above example can be set to
the desired configuration, for instance to create a full C99 version
(parametric arrays):

$> PARGS="-I utilities -DPOLYBENCH_USE_C99_PROTO";
$> for i in `cat utilities/benchmark_list`; do ./utilities/create_cpped_version.sh "$i" "$PARGS"; done

====================================================================================================================================================================================

---------------
* Contact Info: 
---------------

Contacts: Scott Grauer-Gray <sgrauerg@gmail.com>
		  Will Killian <killian@udel.edu>
		  John Cavazos <cavazos@udel.edu>


------------------------
* Paper describing work: 
------------------------

Scott Grauer-Gray, Lifan Xu, Robert Searles, Sudhee Ayalasomayajula, and John Cavazos.  
Auto-tuning a High-Level Language Targeted to GPU Codes. 
To Appear In Proceedings of Innovative Parallel Computing 
(InPar '12), 2012.


Codes are based on PolyBench codes which are able to be parallelized on the GPU; 
Original PolyBench codes available at http://www.cse.ohio-state.edu/~pouchet/software/polybench/.


Acknowledgement:
This work was funded in part by the U.S. National Science Foundation through the NSF
Career award 0953667 and the Defense Advanced Research Projects Agency through the DARPA
Computer Science Study Group (CSSG).
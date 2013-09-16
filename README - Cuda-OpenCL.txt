README

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
* PolyBench/GPU (and other accelerators):  PolyBench Benchmarks on the GPU using CUDA, OpenCL,
* HMPP, and OpenACC.  
* 
* README for CUDA and OpenCL codes
*
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


--------------------------------------
* Available benchmarks in CUDA/OpenCL:
--------------------------------------

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

All the codes (CUDA, OpenCL, HMPP, and OpenACC) are based on PolyBench 3.2.

--------------------------------------------------------------------------------------------
* Instructions - to compile/run CUDA and OpenCL
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


------------------
* Modifying Codes:
------------------

Parameters such as the input sizes, data type, and threshold for GPU-CPU output comparison can be modified using constants
within the codes and .h files.  After modifying, run "make clean" then "make" on relevant code for modifications to take effect in resulting executable.

NOTES ABOUT PARAMETERS:

Input Size:
By default the STANDARD_DATASET as defined in the .cuh/.h file is used as the input size.  The dataset choice can be adjusted from STANDARD_DATASET to other
options (MINI_DATASET, SMALL_DATASET, etc) in the .cuh/.h file, the dataset size can be adjusted by defining the input size manually in the .cuh/.h file, or
the input size can be changed by simply adjusting the STANDARD_DATASET so the program has different input dimensions.

RUN_ON_CPU (in .cu/.c files):
Declares if the kernel will be run on the accelerator and CPU (with the run-time for each given and the outputs compared) or only on the accelerator.  By
default, RUN_ON_CPU is defined so the kernel is run on both the accelerator and the CPU to make it easy to compare accelerator/CPU outputs and run-times.
Commenting out or removing the "#define RUN_ON_CPU" statement and re-compiling the code will cause the kernel to only be run on the accelerator.

DATA_TYPE (in .cuh/.h files):
By default, the DATA_TYPE used in these codes are floats; that can be changed to doubles by changing the DATA_TYPE typedef to "double".  Note that in OpenCL,
the DATA_TYPE needs to be changed in both the .h and .cl files, as the .cl files contain the kernel code and is compiled separately at run-time.

PERCENT_DIFF_ERROR_THRESHOLD (in .cu/.c files):
The PERCENT_DIFF_ERROR_THRESHOLD refers to the percent difference (0.0-100.0) that the GPU and CPU results are allowed to differ and still be considered "matching";
this parameter can be adjusted for each code in the input code file.

OPENCL_DEVICE_SELECTION (in .c files for OpenCL)
Declares the type of accelerator to use for running the OpenCL kernel(s).  By default is set to CL_DEVICE_TYPE_GPU, indicating a GPU accelerator.  
Other options:
CL_DEVICE_TYPE_CPU - run the OpenCL kernel on the CPU
CL_DEVICE_TYPE_ACCELERATOR - run the OpenCL kernel on another accelerator such as the Intel Xeon Phi processor or IBM Cell Blade

------------------------------------
* Other available options:
------------------------------------

These are passed as macro definitions during compilation time 
(e.g -Dname_of_the_option) or can be added with a #define to the code.

- POLYBENCH_STACK_ARRAYS (only applies to allocation on host): 
use stack allocation instead of malloc [default: off]

- POLYBENCH_DUMP_ARRAYS: dump all live-out arrays on stderr [default: off]

- POLYBENCH_CYCLE_ACCURATE_TIMER: Use Time Stamp Counter to monitor
  the execution time of the kernel [default: off]

- MINI_DATASET, SMALL_DATASET, STANDARD_DATASET, LARGE_DATASET,
  EXTRALARGE_DATASET: set the dataset size to be used
  [default: STANDARD_DATASET]

- POLYBENCH_USE_C99_PROTO: Use standard C99 prototype for the functions.

- POLYBENCH_USE_SCALAR_LB: Use scalar loop bounds instead of parametric ones.


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
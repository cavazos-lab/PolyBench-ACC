
# COMPILER OPTIONS -- ACCELERATOR
########################################

# Accelerator Compiler
ACC = roseacc

# Accelerator Compiler flags
ACCFLAGS=--roseacc:desc_format=static_data --roseacc:compile=false

ACC_INC_PATH=`openacc --incpath | tail -n1`
ACC_LIB_PATH=`openacc --libpath | tail -n1`
ACC_LIBS=`openacc --libs | tail -n1`

# COMPILER OPTIONS -- HOST
########################################

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -O2


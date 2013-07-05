# CODE GENERATION OPTIONS
########################################

# Default OpenACC Target is OpenCL
TARGET_LANG = OPENCL

# Uncomment if you want CUDA
# TARGET_LANG = CUDA

# COMPILER OPTIONS -- ACCELERATOR
########################################

# Accelerator Compiler
HMPP = capsmc

# Accelerator Compiler flags
HMPPFLAGS = --codelet-required

# COMPILER OPTIONS -- HOST
########################################

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -O2

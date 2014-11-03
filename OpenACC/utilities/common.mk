INCPATHS = -I$(UTIL_DIR)

BENCHMARK = $(shell basename `pwd`)
EXE = $(BENCHMARK)-seq $(BENCHMARK)-acc
OBJ = rose_$(BENCHMARK).c
SRC = $(BENCHMARK).c
HEADERS = $(BENCHMARK).h

DEPS        := Makefile.dep
DEP_FLAG    := -MM

ACC_EXEC_ENV?=ACC_PROFILING_DB=profile.db

DATASET?=-DSTANDARD_DATASET
DATATYPE?=-DDATA_TYPE=float -DDATA_PRINTF_MODIFIER="\"%0.2f \""
TIMER?=-DPOLYBENCH_TIME

.PHONY: all clean veryclean

all : $(BENCHMARK)-acc

$(OBJ) : $(SRC)
	$(ACC) $(CFLAGS) $(DATASET) $(DATATYPE) $(TIMER) $(ACCFLAGS) $(ACC_INC_PATH) $(INCPATHS) $^

$(BENCHMARK)-acc: $(OBJ) $(BENCHMARK)-data.c $(UTIL_DIR)/polybench.c
	$(CC) -o $@ $(CFLAGS) $(DATASET) $(DATATYPE) $(TIMER) $(ACC_INC_PATH) $(ACC_LIB_PATH) $(INCPATHS) $^ $(ACC_LIBS)

$(BENCHMARK)-seq: $(SRC) $(UTIL_DIR)/polybench.c
	$(CC) -o $@ $(CFLAGS) $(DATASET) $(DATATYPE) $(TIMER) $(INCPATHS) $^ -lm

check: $(BENCHMARK)-acc
	$(ACC_EXEC_ENV) ./$(BENCHMARK)-acc

compare: $(BENCHMARK)-seq $(BENCHMARK)-acc
	./$(BENCHMARK)-seq
	$(ACC_EXEC_ENV) ./$(BENCHMARK)-acc

clean :
	-rm -vf __hmpp* -vf $(EXE) *.sl3 *~ 
	-rm -vf rose_$(BENCHMARK).c $(BENCHMARK)-data.c $(BENCHMARK).cl $(BENCHMARK)

veryclean : clean
	-rm -vf $(DEPS)

$(DEPS): $(SRC) $(HEADERS)
	$(CC) $(INCPATHS) $(DEP_FLAG) $(SRC) > $(DEPS)

-include $(DEPS)

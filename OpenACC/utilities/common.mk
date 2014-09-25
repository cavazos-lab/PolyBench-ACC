INCPATHS = -I$(UTIL_DIR)

BENCHMARK = $(shell basename `pwd`)
EXE = $(BENCHMARK)
OBJ = rose_$(BENCHMARK).o $(BENCHMARK)-data.o
SRC = $(BENCHMARK).c
HEADERS = $(BENCHMARK).h

SRC += $(UTIL_DIR)/polybench.c

DEPS        := Makefile.dep
DEP_FLAG    := -MM

.PHONY: all exe clean veryclean

all : exe

exe : $(EXE)

$(OBJ) : $(SRC)
	$(ACC) $(ACCFLAGS) $(ACC_INC_PATH) $(INCPATHS) $^

$(EXE) : $(OBJ)
	$(CC) -o $@ $(CFLAGS) $(ACC_LIB_PATH) $^ $(ACC_LIBS)

clean :
	-rm -vf __hmpp* -vf $(EXE) *~ 

veryclean : clean
	-rm -vf $(DEPS)

$(DEPS): $(SRC) $(HEADERS)
	$(CC) $(INCPATHS) $(DEP_FLAG) $(SRC) > $(DEPS)

-include $(DEPS)

INCPATHS = -I$(UTIL_DIR)

BENCHMARK = $(shell basename `pwd`)
EXE = $(BENCHMARK)_acc
SRC = $(BENCHMARK).c
HEADERS = $(BENCHMARK).h

SRC += $(UTIL_DIR)/polybench.c

DEPS        := Makefile.dep
DEP_FLAG    := -MM

DATASET ?= STANDARD
DATASET_MACRO := $(DATASET)_DATASET

.PHONY: all exe clean veryclean

all : exe

exe : $(EXE)

$(EXE) : $(SRC)
	$(ACC) $(ACCFLAGS) $(CC) $(CFLAGS) -D${DATASET_MACRO} $(INCPATHS) $^ -o $@

clean :
	-rm -vf __hmpp* -vf $(EXE) *~ 

veryclean : clean
	-rm -vf $(DEPS)

$(DEPS): $(SRC) $(HEADERS)
	$(CC) $(INCPATHS) $(DEP_FLAG) -D${DATASET_MACRO} $(SRC) > $(DEPS)

-include $(DEPS)
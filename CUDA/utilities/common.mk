all:
	nvcc -O3 ${CUFILES} -I${PATH_TO_UTILS} -o ${EXECUTABLE} 
clean:
	rm -f *~ *.exe
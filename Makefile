all: naive_parallel

naive_parallel: ./n-body-sim/src/naive_parallel.c
	gcc ./n-body-sim/src/naive_parallel.c -lOpenCL -lm -o naive_parallel.out
	
clean:
	rm ./n-body-sim/src/naive_parallel.c

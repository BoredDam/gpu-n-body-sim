all: serial

serial: ./src/serial.c
	gcc ./src/serial.c -lOpenCL -lm -O2 -o serial.out
	
clean:
	rm ./src/serial.c

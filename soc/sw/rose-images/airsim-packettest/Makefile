CC=riscv64-unknown-elf-gcc
CFLAGS=-mcmodel=medany -Wall -O2 -fno-common -fno-builtin
LDFLAGS=-static -nostdlib -nostartfiles -lgcc
SOURCES=airsimpackettest.c
OBJECTS=airsimpackettest.o

compile: $(SOURCES)
	$(CC) $(CFLAGS) -c $(SOURCES)
	$(CC) $(LD_FLAGS) $(OBJECTS) -o airsimpackettest.riscv

clean:
	 rm -f *.o *.riscv *.txt
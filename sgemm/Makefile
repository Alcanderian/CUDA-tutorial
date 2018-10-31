CC=icpc
NVCC=nvcc
LD=nvcc
TARGET=sample

CFLAGS= -O3 -qopenmp -std=c++11
CUFLAGS= -O3 -arch=sm_37 -ccbin=$(CC) -std=c++11 -Xcompiler -qopenmp
LDFLAGS= -arch=sm_37 -ccbin=$(CC) -std=c++11 -Xcompiler -qopenmp

CPOBJS= sample.o
CUOBJS= interfaces.o

all:build

build:$(TARGET)

$(TARGET): $(CPOBJS) $(CUOBJS)
	$(LD) $(LDFLAGS) $(CPOBJS) $(CUOBJS) -o $(TARGET)
	rm -rf $(CPOBJS) $(CUOBJS)

$(CPOBJS): %.o: %.cpp
	$(CC) $(CFLAGS) -c $<

$(CUOBJS): %.o: %.cu
	$(NVCC) $(CUFLAGS) -c $<

clean:
	rm -rf $(CPOBJS) $(CUOBJS)

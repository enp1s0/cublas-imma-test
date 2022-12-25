NVCC=nvcc
NVCCFLAGS=-std=c++17
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80
NVCCFLAGS+=-I./src/cutf/include -lcublas

TARGET=cublas-imma.test

$(TARGET):src/main.cu
	$(NVCC) $< -o $@ $(NVCCFLAGS)
  
clean:
	rm -f $(TARGET)

NVCC ?= nvcc
TARGET := cuda_simplified_ops
SRC := cuda_simplified_ops.cu

.PHONY: all run-cuda clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -O2 -arch=native $(SRC) -o $(TARGET)

run-cuda: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

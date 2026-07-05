NVCC ?= nvcc
TARGET := cuda_ops_intro
SRC := cuda_ops_intro.cu

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -O2 -arch=native $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

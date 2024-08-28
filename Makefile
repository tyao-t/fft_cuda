SRC_DIR = src
BIN_DIR = bin

SRC_FILE = fft.cu
OUTPUT_FILE = fft

NVCC = nvcc
NVCC_FLAGS = -O3 -lcublas -UENABLE_DEBUG

all: $(BIN_DIR)/$(OUTPUT_FILE)

$(BIN_DIR)/$(OUTPUT_FILE): $(SRC_DIR)/$(SRC_FILE)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
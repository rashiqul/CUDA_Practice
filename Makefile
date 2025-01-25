# Variables
BUILD_DIR = build
TARGET = cuda_app

# Commands
.PHONY: all clean build run

all: build

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. && make
	@echo "Build complete. Executable is in the $(BUILD_DIR) directory."

clean:
	@rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory."

run: build
	@echo "Running the CUDA application..."
	@$(BUILD_DIR)/$(TARGET)

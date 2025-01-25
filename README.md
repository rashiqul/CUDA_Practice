# CUDA Practice
A repository to practice CUDA programming using CMake. This repository demonstrates how to set up, build, and run CUDA applications efficiently with a clean and straightforward workflow.

## Device Specifications
- **GPU**: NVIDIA Quadro T1000
- **CUDA Version**: 12.8
- **C++ Compiler**: GCC 11.4.0
- **CMake Version**: 3.18+
- **OS**: Windows Subsystem for Linux (WSL)

## Directory Structure
workspace/ 
    ├── src/ 
        │ └── main.cu  # CUDA application code 
    ├── build/         # Directory for build files (auto-generated) 
    ├── CMakeLists.txt # CMake configuration file 
    ├── Makefile       # Makefile for building and cleaning

## Variables in Makefile
- `BUILD_DIR`: Directory where build files are generated (`build/`).
- `TARGET`: The executable name (`cuda_app`).

## Commands in Makefile
### `.PHONY` Targets
- `all`: Default target, builds the project.
- `build`: Generates the build directory, runs CMake, and compiles the project.
- `clean`: Cleans up the build directory and its contents.
- `run`: Builds and runs the CUDA application.

### Usage
#### Build the Project
```bash
make build
make run
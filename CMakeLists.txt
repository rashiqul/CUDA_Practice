cmake_minimum_required(VERSION 3.18)
project(CUDA_Project LANGUAGES CXX CUDA)

# Set output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Enable CUDA support
enable_language(CUDA)

# Add the source file
file(GLOB SOURCE_FILES src/*.cu)

# Iterate over each source file and create an executable
foreach(SOURCE_FILE ${SOURCE_FILES})
    get_filename_component(EXECUTABLE_NAME ${SOURCE_FILE} NAME_WE)
    add_executable(${EXECUTABLE_NAME} ${SOURCE_FILE})

    set_target_properties(${EXECUTABLE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()
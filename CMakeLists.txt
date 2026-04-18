cmake_minimum_required(VERSION 3.18)
project(CUDA_Project LANGUAGES CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

enable_language(CUDA)

# Device-side debug info (-G) + host debug info (-g) when building Debug
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

# Add the source file
file(GLOB SOURCE_FILES src/*.cu)

# Iterate over each source file and create an executable
foreach(SOURCE_FILE ${SOURCE_FILES})
    get_filename_component(EXECUTABLE_NAME ${SOURCE_FILE} NAME_WE)
    add_executable(${EXECUTABLE_NAME} ${SOURCE_FILE})

    set_target_properties(${EXECUTABLE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()
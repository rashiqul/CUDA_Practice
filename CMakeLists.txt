cmake_minimum_required(VERSION 3.18)
project(CUDA_Project LANGUAGES CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_language(CUDA)

# Device-side debug info (-G) + host debug info (-g) when building Debug
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

add_subdirectory(src)

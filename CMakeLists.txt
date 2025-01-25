cmake_minimum_required(VERSION 3.18)
project(CUDA_Project LANGUAGES CXX CUDA)

# Set output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Enable CUDA support
enable_language(CUDA)

# Add the source file
set(SOURCE_FILES src/vector_add.cu)

# Define the executable
add_executable(cuda_app ${SOURCE_FILES})

# Set compiler flags for CUDA (optional)
set_target_properties(cuda_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Post-build commands
add_custom_command(
    TARGET cuda_app
    POST_BUILD
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cuda_app > ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/output.log 2> ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/error.log && cat ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/output.log
    COMMENT "Running CUDA application and displaying/logging output."
)


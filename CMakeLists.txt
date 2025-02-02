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

    # Set compiler flags for CUDA (optional)
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Post-build commands
    add_custom_command(
        TARGET ${EXECUTABLE_NAME}
        POST_BUILD
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${EXECUTABLE_NAME} > ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${EXECUTABLE_NAME}_output.log 2> ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${EXECUTABLE_NAME}_error.log && echo ""
        COMMENT "Running ${EXECUTABLE_NAME} and logging output."
    )
endforeach()
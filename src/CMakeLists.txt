file(GLOB SOURCE_FILES */*.cu)

foreach(SOURCE_FILE ${SOURCE_FILES})
    get_filename_component(EXECUTABLE_NAME ${SOURCE_FILE} NAME_WE)
    get_filename_component(SOURCE_DIR ${SOURCE_FILE} DIRECTORY)

    # Include any companion .cpp or .c files in the same directory
    file(GLOB COMPANION_FILES "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.c")

    add_executable(${EXECUTABLE_NAME} ${SOURCE_FILE} ${COMPANION_FILES})
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    if(TARGET libwb)
        target_link_libraries(${EXECUTABLE_NAME} libwb)
    endif()
endforeach()

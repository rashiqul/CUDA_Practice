file(GLOB SOURCE_FILES */*.cu)

set(PROCESSED_DIRS "")

foreach(SOURCE_FILE ${SOURCE_FILES})
    get_filename_component(SOURCE_DIR ${SOURCE_FILE} DIRECTORY)

    # Skip directories we've already handled
    if(NOT "${SOURCE_DIR}" IN_LIST PROCESSED_DIRS)
        list(APPEND PROCESSED_DIRS ${SOURCE_DIR})

        file(GLOB CU_FILES "${SOURCE_DIR}/*.cu")
        file(GLOB COMPANION_FILES "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.c")

        list(LENGTH CU_FILES CU_COUNT)
        if(CU_COUNT EQUAL 1)
            # Single .cu file: name the executable after the file
            get_filename_component(EXECUTABLE_NAME ${SOURCE_FILE} NAME_WE)
        else()
            # Multiple .cu files: name the executable after the directory
            get_filename_component(EXECUTABLE_NAME ${SOURCE_DIR} NAME)
        endif()

        add_executable(${EXECUTABLE_NAME} ${CU_FILES} ${COMPANION_FILES})
        set_target_properties(${EXECUTABLE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        if(TARGET libwb)
            target_link_libraries(${EXECUTABLE_NAME} libwb)
        endif()
    endif()
endforeach()

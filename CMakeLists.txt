cmake_minimum_required(VERSION 3.18)
project(CUDA_Project LANGUAGES CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_language(CUDA)

# Device-side debug info (-G) + host debug info (-g) when building Debug
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

# ── libwb (pre-built once via `make configure` / `build.ps1 configure`) ─────────
set(LIBWB_DIR "${CMAKE_SOURCE_DIR}/extern/libwb")

# Filename differs by platform: .a on Unix/WSL, .lib on Windows (MSVC/Ninja)
if(WIN32)
    set(LIBWB_LIB "${LIBWB_DIR}/build/wb.lib")
else()
    set(LIBWB_LIB "${LIBWB_DIR}/build/libwb.a")
endif()

if(EXISTS "${LIBWB_LIB}")
    add_library(libwb STATIC IMPORTED GLOBAL)
    set_target_properties(libwb PROPERTIES
        IMPORTED_LOCATION "${LIBWB_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${LIBWB_DIR}"
        INTERFACE_COMPILE_DEFINITIONS "WB_USE_COURSERA"
    )
    message(STATUS "libwb found: ${LIBWB_LIB}")
else()
    message(STATUS "libwb not built yet — run `make configure` (Linux/WSL) or `.\\build.ps1 configure` (Windows) first")
endif()

# ── Flatten multi-config output (Visual Studio / Ninja Multi-Config) ──────────
# Ensures binaries always land in CMAKE_BINARY_DIR rather than
# CMAKE_BINARY_DIR/Debug or CMAKE_BINARY_DIR/Release.
foreach(_cfg Release Debug RelWithDebInfo MinSizeRel)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_cfg} ${CMAKE_BINARY_DIR})
endforeach()

add_subdirectory(src)

cmake_minimum_required(VERSION 3.18)

# CUDA architecture — must be set before project() so cmake uses it during CUDA
# language initialization. Targets Quadro T1000 (Turing = sm_75).
# Override at configure time: cmake -DCMAKE_CUDA_ARCHITECTURES=86 ...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project(CUDA_Project LANGUAGES CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Prevent nvcc from using response files for includes so that cpptools
# (VS Code IntelliSense) can read include paths directly from compile_commands.json.
# Only applied on non-Windows: Windows has an ~8191-char command-line limit and
# relies on response files to stay under it.
if(NOT WIN32)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
endif()

# Always use the release CRT (/MD) so we match the pre-built libwb (Release).
# Debug symbols are still emitted via CMAKE_CUDA_FLAGS_DEBUG (-g -G).
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

enable_language(CUDA)

# Device-side debug info (-G) + host debug info (-g) when building Debug
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

# ── libwb (pre-built once via `make configure` / `build.ps1 configure`) ─────────
set(LIBWB_DIR "${CMAKE_SOURCE_DIR}/extern/libwb")

# Filename differs by platform: .a on Unix/WSL, .lib on Windows (MSVC/Ninja).
# Visual Studio generator puts the lib in a config subfolder (Release/Debug).
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

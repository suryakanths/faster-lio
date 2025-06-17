cmake_minimum_required(VERSION 3.5)
project(faster_lio)

set(DEFAULT_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS}")

# Enable CUDA support
option(USE_CUDA "Enable CUDA acceleration" ON)
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    
    if(CUDA_FOUND)
        message(STATUS "CUDA found: ${CUDA_VERSION}")
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_EXTENSIONS OFF)
        
        # Detect Jetson platform and set appropriate CUDA architectures
        if(EXISTS "/proc/device-tree/model")
            file(READ "/proc/device-tree/model" JETSON_MODEL)
            string(FIND "${JETSON_MODEL}" "NVIDIA Jetson" IS_JETSON)
            if(NOT IS_JETSON EQUAL -1)
                message(STATUS "Detected Jetson platform: ${JETSON_MODEL}")
                # Jetson CUDA architectures (covers most Jetson devices)
                # Xavier NX/AGX: 7.2, Orin: 8.7, Nano: 5.3, TX2: 6.2
                set(CMAKE_CUDA_ARCHITECTURES "53;62;72;87")
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -lineinfo --ptxas-options=-v")
                # Jetson-specific optimizations
                set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --default-stream per-thread")
            else()
                # Desktop GPU architectures (RTX 4060, etc.)
                set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -lineinfo")
                set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
            endif()
        else()
            # Fallback for non-ARM systems (assume desktop)
            set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -lineinfo")
            set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
        endif()
        
        # Add compile definition to avoid PCL template issues in CUDA
        add_compile_definitions(CUDA_NVCC_COMPILATION=1)
        
        add_definitions(-DUSE_CUDA)
        message(STATUS "CUDA acceleration enabled")
    else()
        message(WARNING "CUDA not found, falling back to CPU-only mode")
        set(USE_CUDA OFF)
    endif()
endif()

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")
add_subdirectory(thirdparty/livox_ros_driver)

include(cmake/packages.cmake)

#definitions
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

# options
option(WITH_IVOX_NODE_TYPE_PHC "Use PHC instead of default ivox node" OFF)

if (WITH_IVOX_NODE_TYPE_PHC)
    message("USING_IVOX_NODE_TYPE_PHC")
    add_definitions(-DIVOX_NODE_TYPE_PHC)
else ()
    message("USING_IVOX_NODE_TYPE_DEFAULT")
endif()

add_subdirectory(src)
add_subdirectory(app)

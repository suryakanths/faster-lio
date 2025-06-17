# Determine source files based on CUDA availability
set(FASTER_LIO_SOURCES
    laser_mapping.cc
    pointcloud_preprocess.cc
    options.cc
    utils.cc
    cuda_utils_cpu.cc
    jetson_optimizations.cc
)

# Add CUDA source files if CUDA is enabled
if(USE_CUDA AND CUDA_FOUND)
    list(APPEND FASTER_LIO_SOURCES cuda_utils.cu cuda_utils_impl.cu cuda_utils_wrapper.cc)
    # Enable CUDA language for this directory
    enable_language(CUDA)
endif()

add_library(${PROJECT_NAME} ${FASTER_LIO_SOURCES})

add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencpp livox_ros_driver_gencpp)

# Base libraries
set(FASTER_LIO_LIBS
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${PYTHON_LIBRARIES}
    tbb
    glog
    yaml-cpp
)

# Add CUDA libraries if enabled
if(USE_CUDA AND CUDA_FOUND)
    list(APPEND FASTER_LIO_LIBS
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${CUDA_CUFFT_LIBRARIES}
    )
    
    # Set CUDA properties for the library (CUDA 10.1 supports up to C++14)
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_STANDARD 14)
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_STANDARD_REQUIRED ON)
    
    # Set specific compile flags for CUDA files to avoid PCL template issues
    set_source_files_properties(cuda_utils_impl.cu PROPERTIES 
        COMPILE_FLAGS "-DCUDA_SEPARABLE_COMPILATION=ON"
    )
endif()

target_link_libraries(${PROJECT_NAME} ${FASTER_LIO_LIBS})

target_include_directories(${PROJECT_NAME} PRIVATE ${PYTHON_INCLUDE_DIRS})

# Add CUDA include directories if enabled
if(USE_CUDA AND CUDA_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
endif()

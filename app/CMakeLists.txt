add_executable(run_mapping_online run_mapping_online.cc)
target_link_libraries(run_mapping_online
        ${PROJECT_NAME} gflags
        )

add_executable(run_mapping_offline run_mapping_offline.cc)
target_link_libraries(run_mapping_offline
        ${PROJECT_NAME} gflags
        )

# Set CUDA properties for executables if CUDA is enabled (CUDA 10.1 supports up to C++14)
if(USE_CUDA AND CUDA_FOUND)
    set_property(TARGET run_mapping_online PROPERTY CUDA_STANDARD 14)
    set_property(TARGET run_mapping_offline PROPERTY CUDA_STANDARD 14)
endif()

// Minimal CUDA file - Only contains CUDA runtime code, no PCL headers
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <iostream>

#ifdef USE_CUDA

namespace faster_lio {

// This file is now empty - all PCL-related code moved to cuda_utils_wrapper.cc
// The actual CUDA kernels are in cuda_utils_impl.cu
// This separation allows us to avoid PCL template compilation issues with NVCC

// Simple CUDA utility functions that don't depend on PCL
namespace cuda_utils {

bool CheckCudaError(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        LOG(ERROR) << "CUDA error in " << operation << ": " << cudaGetErrorString(error);
        return false;
    }
    return true;
}

int GetCudaDeviceCount() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        return 0;
    }
    return device_count;
}

bool SetCudaDevice(int device_id) {
    cudaError_t error = cudaSetDevice(device_id);
    return CheckCudaError(error, "cudaSetDevice");
}

void PrintCudaDeviceInfo() {
    int device_count = GetCudaDeviceCount();
    LOG(INFO) << "Found " << device_count << " CUDA devices";
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaError_t error = cudaGetDeviceProperties(&prop, i);
        if (error == cudaSuccess) {
            LOG(INFO) << "Device " << i << ": " << prop.name 
                      << " (Compute " << prop.major << "." << prop.minor << ")";
        }
    }
}

} // namespace cuda_utils
} // namespace faster_lio

#endif // USE_CUDA

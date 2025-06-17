/**
 * @file jetson_optimizations.cc
 * @brief Implementation of Jetson-specific optimizations
 */

#include "jetson_optimizations.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <glog/logging.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace faster_lio {
namespace jetson {

JetsonOptimizer::JetsonOptimizer() {}

JetsonInfo JetsonOptimizer::DetectJetsonPlatform() {
    JetsonInfo info;
    
    // Read Jetson model from device tree
    info.model = ReadJetsonModel();
    info.is_jetson = info.model.find("NVIDIA Jetson") != std::string::npos;
    
    if (info.is_jetson) {
        LOG(INFO) << "Detected Jetson platform: " << info.model;
        ParseCudaCapabilities(info);
    }
    
    return info;
}

std::string JetsonOptimizer::ReadJetsonModel() {
    std::ifstream model_file("/proc/device-tree/model");
    if (!model_file.is_open()) {
        return "Unknown";
    }
    
    std::string model;
    std::getline(model_file, model);
    
    // Remove null terminator if present
    if (!model.empty() && model.back() == '\0') {
        model.pop_back();
    }
    
    return model;
}

bool JetsonOptimizer::ParseCudaCapabilities(JetsonInfo& info) {
#ifdef USE_CUDA
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        LOG(WARNING) << "No CUDA devices found";
        return false;
    }
    
    cudaDeviceProp prop;
    error = cudaGetDeviceProperties(&prop, 0);
    if (error != cudaSuccess) {
        LOG(ERROR) << "Failed to get CUDA device properties";
        return false;
    }
    
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.total_memory_mb = prop.totalGlobalMem / (1024 * 1024);
    info.shared_memory_per_block = prop.sharedMemPerBlock;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.supports_unified_memory = prop.unifiedAddressing;
    
    LOG(INFO) << "CUDA Device: " << prop.name;
    LOG(INFO) << "Compute Capability: " << prop.major << "." << prop.minor;
    LOG(INFO) << "Total Memory: " << info.total_memory_mb << " MB";
    LOG(INFO) << "Unified Memory: " << (info.supports_unified_memory ? "Yes" : "No");
    
    return true;
#else
    LOG(WARNING) << "CUDA not enabled";
    return false;
#endif
}

JetsonKernelConfig JetsonOptimizer::GetOptimalKernelConfig(const JetsonInfo& info) {
    JetsonKernelConfig config;
    
    if (!info.is_jetson) {
        return config; // Use defaults for non-Jetson platforms
    }
    
    // Configure based on compute capability and model
    int compute_version = info.compute_capability_major * 10 + info.compute_capability_minor;
    
    switch (compute_version) {
        case 53: // Jetson Nano
            config.block_size = models::NANO_OPTIMAL_BLOCK_SIZE;
            config.memory_pool_fraction = models::NANO_MEMORY_FRACTION;
            config.use_unified_memory = true; // Helps with limited memory
            config.enable_concurrent_execution = false; // Single SM
            break;
            
        case 62: // Jetson TX2
            config.block_size = models::TX2_OPTIMAL_BLOCK_SIZE;
            config.memory_pool_fraction = models::TX2_MEMORY_FRACTION;
            config.use_unified_memory = true;
            config.enable_concurrent_execution = true;
            break;
            
        case 72: // Jetson Xavier NX/AGX
            config.block_size = models::XAVIER_OPTIMAL_BLOCK_SIZE;
            config.memory_pool_fraction = models::XAVIER_MEMORY_FRACTION;
            config.use_unified_memory = true;
            config.enable_concurrent_execution = true;
            config.grid_size_multiplier = 2; // More SMs available
            break;
            
        case 87: // Jetson Orin
            config.block_size = models::ORIN_OPTIMAL_BLOCK_SIZE;
            config.memory_pool_fraction = models::ORIN_MEMORY_FRACTION;
            config.use_unified_memory = true;
            config.enable_concurrent_execution = true;
            config.grid_size_multiplier = 4; // Most powerful Jetson
            break;
            
        default:
            LOG(WARNING) << "Unknown Jetson compute capability: " << compute_version;
            break;
    }
    
    // Adjust for memory constraints
    if (info.total_memory_mb < 4000) { // Less than 4GB
        config.memory_pool_fraction *= 0.8f; // Be more conservative
        config.block_size = std::min(config.block_size, 256);
    }
    
    LOG(INFO) << "Jetson kernel config - Block size: " << config.block_size
              << ", Memory fraction: " << config.memory_pool_fraction
              << ", Unified memory: " << config.use_unified_memory;
    
    return config;
}

bool JetsonOptimizer::InitializeJetsonCuda(const JetsonInfo& info) {
#ifdef USE_CUDA
    if (!info.is_jetson) {
        return true; // No special initialization needed for non-Jetson
    }
    
    // Set device
    cudaError_t error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        LOG(ERROR) << "Failed to set CUDA device: " << cudaGetErrorString(error);
        return false;
    }
    
    // Configure memory pool if supported
    if (info.total_memory_mb > 0) {
        size_t free_memory, total_memory;
        error = cudaMemGetInfo(&free_memory, &total_memory);
        if (error == cudaSuccess) {
            LOG(INFO) << "Available GPU memory: " << free_memory / (1024*1024) << " MB";
        }
    }
    
    // Enable unified memory hints for Jetson
    if (info.supports_unified_memory) {
        LOG(INFO) << "Enabling unified memory optimizations for Jetson";
        // Set memory advice for better performance
        // This would be done per allocation in practice
    }
    
    return true;
#else
    return false;
#endif
}

bool JetsonOptimizer::CheckThermalThrottling() {
    // Check thermal zones common on Jetson platforms
    const std::vector<std::string> thermal_zones = {
        "thermal_zone0", "thermal_zone1", "thermal_zone2", "CPU-therm", "GPU-therm"
    };
    
    bool throttling_detected = false;
    for (const auto& zone : thermal_zones) {
        float temp = ReadThermalZoneTemp(zone);
        if (temp > 80.0f) { // Typical throttling threshold
            LOG(WARNING) << "High temperature detected in " << zone << ": " << temp << "°C";
            throttling_detected = true;
        }
    }
    
    return throttling_detected;
}

float JetsonOptimizer::ReadThermalZoneTemp(const std::string& zone) {
    std::string temp_path = "/sys/class/thermal/" + zone + "/temp";
    std::ifstream temp_file(temp_path);
    
    if (!temp_file.is_open()) {
        return 0.0f; // Zone doesn't exist
    }
    
    int temp_millidegrees;
    temp_file >> temp_millidegrees;
    
    return static_cast<float>(temp_millidegrees) / 1000.0f;
}

int JetsonOptimizer::GetCurrentPowerMode() {
    std::ifstream nvpmodel_file("/etc/nvpmodel.conf");
    if (!nvpmodel_file.is_open()) {
        return -1; // Not a Jetson or nvpmodel not available
    }
    
    // This is a simplified implementation
    // In practice, you'd parse the nvpmodel configuration
    return 0; // Default power mode
}

size_t JetsonOptimizer::GetOptimalBatchSize(const JetsonInfo& info, size_t point_size) {
    if (!info.is_jetson) {
        return 100000; // Default for desktop GPUs
    }
    
    // Calculate based on available memory and point size
    size_t available_memory = static_cast<size_t>(info.total_memory_mb * 1024 * 1024 * 0.7); // 70% of total
    size_t max_points = available_memory / (point_size * 4); // 4 float components per point
    
    // Apply model-specific constraints
    int compute_version = info.compute_capability_major * 10 + info.compute_capability_minor;
    switch (compute_version) {
        case 53: // Nano - be conservative
            max_points = std::min(max_points, static_cast<size_t>(50000));
            break;
        case 62: // TX2
            max_points = std::min(max_points, static_cast<size_t>(100000));
            break;
        case 72: // Xavier
            max_points = std::min(max_points, static_cast<size_t>(200000));
            break;
        case 87: // Orin
            max_points = std::min(max_points, static_cast<size_t>(500000));
            break;
        default:
            max_points = std::min(max_points, static_cast<size_t>(100000));
            break;
    }
    
    return max_points;
}

bool JetsonOptimizer::ConfigureUnifiedMemory(bool enable) {
#ifdef USE_CUDA
    if (enable) {
        // Query if unified memory is supported
        int unified_addressing = 0;
        cudaDeviceGetAttribute(&unified_addressing, cudaDevAttrUnifiedAddressing, 0);
        
        if (unified_addressing) {
            LOG(INFO) << "Unified memory is supported and enabled";
            return true;
        } else {
            LOG(WARNING) << "Unified memory not supported on this device";
            return false;
        }
    }
    return true;
#else
    return false;
#endif
}

} // namespace jetson
} // namespace faster_lio

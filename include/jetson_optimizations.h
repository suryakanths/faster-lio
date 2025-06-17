/**
 * @file jetson_optimizations.h
 * @brief Jetson-specific optimizations for faster-lio CUDA implementation
 * @author ROS C++ Systems Architect
 * 
 * This header provides Jetson-specific optimizations for real-time SLAM:
 * - Memory management for unified memory architecture
 * - Power-aware CUDA kernel tuning
 * - Thermal throttling detection
 * - Performance scaling based on Jetson model
 */

#ifndef FASTER_LIO_JETSON_OPTIMIZATIONS_H
#define FASTER_LIO_JETSON_OPTIMIZATIONS_H

#include <string>
#include <memory>
#include <cstring>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace faster_lio {
namespace jetson {

/**
 * @brief Jetson device information and capabilities
 */
struct JetsonInfo {
    bool is_jetson;
    std::string model;
    int compute_capability_major;
    int compute_capability_minor;
    size_t total_memory_mb;
    size_t shared_memory_per_block;
    int max_threads_per_block;
    bool supports_unified_memory;
    
    JetsonInfo() : is_jetson(false), compute_capability_major(0), 
                   compute_capability_minor(0), total_memory_mb(0),
                   shared_memory_per_block(0), max_threads_per_block(0),
                   supports_unified_memory(false) {}
};

/**
 * @brief Jetson-optimized CUDA kernel configuration
 */
struct JetsonKernelConfig {
    int block_size;
    int grid_size_multiplier;
    bool use_unified_memory;
    bool enable_concurrent_execution;
    float memory_pool_fraction;  // Fraction of GPU memory to pre-allocate
    
    JetsonKernelConfig() : block_size(256), grid_size_multiplier(1),
                          use_unified_memory(false), enable_concurrent_execution(true),
                          memory_pool_fraction(0.7f) {}
};

/**
 * @brief Jetson platform detection and optimization manager
 */
class JetsonOptimizer {
public:
    JetsonOptimizer();
    ~JetsonOptimizer() = default;
    
    /**
     * @brief Detect if running on Jetson and get device capabilities
     */
    static JetsonInfo DetectJetsonPlatform();
    
    /**
     * @brief Get optimized kernel configuration for detected Jetson
     */
    static JetsonKernelConfig GetOptimalKernelConfig(const JetsonInfo& info);
    
    /**
     * @brief Initialize CUDA with Jetson-specific optimizations
     */
    static bool InitializeJetsonCuda(const JetsonInfo& info);
    
    /**
     * @brief Monitor thermal state and adjust performance
     * @return true if thermal throttling detected
     */
    static bool CheckThermalThrottling();
    
    /**
     * @brief Get current power mode (for Jetson power management)
     */
    static int GetCurrentPowerMode();
    
    /**
     * @brief Calculate optimal point cloud batch size for Jetson memory
     */
    static size_t GetOptimalBatchSize(const JetsonInfo& info, size_t point_size);
    
    /**
     * @brief Enable/disable CUDA unified memory for Jetson
     */
    static bool ConfigureUnifiedMemory(bool enable);
    
private:
    static std::string ReadJetsonModel();
    static bool ParseCudaCapabilities(JetsonInfo& info);
    static float ReadThermalZoneTemp(const std::string& zone);
};

/**
 * @brief RAII wrapper for Jetson-optimized CUDA memory management
 */
template<typename T>
class JetsonManagedMemory {
public:
    explicit JetsonManagedMemory(size_t count, bool use_unified = false) 
        : size_(count * sizeof(T)), ptr_(nullptr), unified_(use_unified) {
        Allocate();
    }
    
    ~JetsonManagedMemory() {
        Deallocate();
    }
    
    // Non-copyable, movable
    JetsonManagedMemory(const JetsonManagedMemory&) = delete;
    JetsonManagedMemory& operator=(const JetsonManagedMemory&) = delete;
    
    JetsonManagedMemory(JetsonManagedMemory&& other) noexcept 
        : size_(other.size_), ptr_(other.ptr_), unified_(other.unified_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    T* get() const { return ptr_; }
    size_t size() const { return size_; }
    bool is_unified() const { return unified_; }
    
    /**
     * @brief Copy data to device (no-op if unified memory)
     */
    bool CopyToDevice(const T* host_data, size_t count) {
#ifdef USE_CUDA
        if (unified_) {
            std::memcpy(ptr_, host_data, count * sizeof(T));
            return true;
        } else {
            cudaError_t error = cudaMemcpy(ptr_, host_data, count * sizeof(T), cudaMemcpyHostToDevice);
            return error == cudaSuccess;
        }
#else
        return false;
#endif
    }
    
    /**
     * @brief Copy data from device (no-op if unified memory)
     */
    bool CopyFromDevice(T* host_data, size_t count) {
#ifdef USE_CUDA
        if (unified_) {
            std::memcpy(host_data, ptr_, count * sizeof(T));
            return true;
        } else {
            cudaError_t error = cudaMemcpy(host_data, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost);
            return error == cudaSuccess;
        }
#else
        return false;
#endif
    }
    
private:
    void Allocate() {
#ifdef USE_CUDA
        if (unified_) {
            cudaError_t error = cudaMallocManaged(&ptr_, size_);
            if (error != cudaSuccess) {
                ptr_ = nullptr;
            }
        } else {
            cudaError_t error = cudaMalloc(&ptr_, size_);
            if (error != cudaSuccess) {
                ptr_ = nullptr;
            }
        }
#endif
    }
    
    void Deallocate() {
#ifdef USE_CUDA
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
#endif
    }
    
    size_t size_;
    T* ptr_;
    bool unified_;
};

// Jetson model-specific optimizations
namespace models {
    // Jetson Nano (compute capability 5.3)
    constexpr int NANO_OPTIMAL_BLOCK_SIZE = 128;
    constexpr float NANO_MEMORY_FRACTION = 0.6f;
    
    // Jetson TX2 (compute capability 6.2)
    constexpr int TX2_OPTIMAL_BLOCK_SIZE = 256;
    constexpr float TX2_MEMORY_FRACTION = 0.7f;
    
    // Jetson Xavier NX/AGX (compute capability 7.2)
    constexpr int XAVIER_OPTIMAL_BLOCK_SIZE = 512;
    constexpr float XAVIER_MEMORY_FRACTION = 0.8f;
    
    // Jetson Orin (compute capability 8.7)
    constexpr int ORIN_OPTIMAL_BLOCK_SIZE = 512;
    constexpr float ORIN_MEMORY_FRACTION = 0.85f;
}

} // namespace jetson
} // namespace faster_lio

#endif // FASTER_LIO_JETSON_OPTIMIZATIONS_H

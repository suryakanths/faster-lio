// PCL interface wrapper - compiled only with C++, not NVCC
#include <glog/logging.h>
#include "cuda_utils.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

#ifdef USE_CUDA

namespace faster_lio {
namespace cuda_impl {
// External C interface to pure CUDA kernels
extern "C" {
    cudaError_t cuda_filter_points_by_distance(const float* d_points_x,
                                              const float* d_points_y,
                                              const float* d_points_z,
                                              const float* d_points_intensity,
                                              int* d_valid_flags,
                                              float min_distance,
                                              float max_distance,
                                              int num_points);
    
    cudaError_t cuda_transform_point_cloud(const float* d_input_x,
                                          const float* d_input_y,
                                          const float* d_input_z,
                                          const float* d_input_intensity,
                                          float* d_output_x,
                                          float* d_output_y,
                                          float* d_output_z,
                                          float* d_output_intensity,
                                          const float* d_rotation_matrix,
                                          const float* d_translation_vector,
                                          int num_points);
    
    cudaError_t cuda_downsample_point_cloud(const float* d_input_x,
                                           const float* d_input_y,
                                           const float* d_input_z,
                                           const float* d_input_intensity,
                                           float* d_output_x,
                                           float* d_output_y,
                                           float* d_output_z,
                                           float* d_output_intensity,
                                           const int* d_indices,
                                           int num_output_points);
    
    // PGO correction functions
    cudaError_t cuda_apply_pgo_corrections(const float* d_input_x,
                                         const float* d_input_y,
                                         const float* d_input_z,
                                         const float* d_input_intensity,
                                         const float* d_timestamps,
                                         float* d_output_x,
                                         float* d_output_y,
                                         float* d_output_z,
                                         float* d_output_intensity,
                                         const float* d_original_poses,
                                         const float* d_corrected_poses,
                                         const float* d_pose_timestamps,
                                         int num_poses,
                                         int num_points);
    
    cudaError_t cuda_assign_timestamps(float* d_timestamps,
                                     const int* d_scan_indices,
                                     const float* d_scan_timestamps,
                                     int num_points);
}
} // namespace cuda_impl

// Implementation of CudaPointCloudProcessor using the external C interface
struct CudaPointCloudProcessor::CudaData {
    float* d_points_x;
    float* d_points_y;
    float* d_points_z;
    float* d_points_intensity;
    float* d_rotation_matrix;
    float* d_translation_vector;
    float* d_output_x;
    float* d_output_y;
    float* d_output_z;
    float* d_output_intensity;
    int* d_valid_flags;
    size_t allocated_size;
    
    CudaData() : d_points_x(nullptr), d_points_y(nullptr), d_points_z(nullptr),
                 d_points_intensity(nullptr), d_rotation_matrix(nullptr), 
                 d_translation_vector(nullptr), d_output_x(nullptr), 
                 d_output_y(nullptr), d_output_z(nullptr), 
                 d_output_intensity(nullptr), d_valid_flags(nullptr),
                 allocated_size(0) {}
};

CudaPointCloudProcessor::CudaPointCloudProcessor() 
    : cuda_data_(std::make_unique<CudaData>()), allocated_points_(0), cuda_initialized_(false) {
    // Detect Jetson platform and apply optimizations
    jetson_info_ = jetson::JetsonOptimizer::DetectJetsonPlatform();
    jetson_config_ = jetson::JetsonOptimizer::GetOptimalKernelConfig(jetson_info_);
    
    cuda_initialized_ = InitializeCuda();
}

CudaPointCloudProcessor::~CudaPointCloudProcessor() {
    CleanupCuda();
}

bool CudaPointCloudProcessor::IsCudaAvailable() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
}

bool CudaPointCloudProcessor::InitializeCuda() {
    if (!IsCudaAvailable()) {
        LOG(WARNING) << "CUDA not available";
        return false;
    }
    
    // Apply Jetson-specific initialization
    if (!jetson::JetsonOptimizer::InitializeJetsonCuda(jetson_info_)) {
        LOG(ERROR) << "Failed to initialize Jetson CUDA optimizations";
        return false;
    }
    
    // Check for thermal throttling on Jetson platforms
    if (jetson_info_.is_jetson && jetson::JetsonOptimizer::CheckThermalThrottling()) {
        LOG(WARNING) << "Thermal throttling detected - performance may be reduced";
    }
    
    LOG(INFO) << "CUDA initialized successfully";
    if (jetson_info_.is_jetson) {
        LOG(INFO) << "Jetson optimizations enabled - Model: " << jetson_info_.model;
        LOG(INFO) << "Optimal batch size: " << jetson::JetsonOptimizer::GetOptimalBatchSize(jetson_info_, sizeof(PointType));
    }
    
    return true;
}

void CudaPointCloudProcessor::CleanupCuda() {
    DeallocateGpuMemory();
}

bool CudaPointCloudProcessor::AllocateGpuMemory(size_t num_points) {
    if (num_points <= allocated_points_) {
        return true; // Already allocated enough memory
    }
    
    // Deallocate existing memory
    DeallocateGpuMemory();
    
    size_t float_size = num_points * sizeof(float);
    size_t int_size = num_points * sizeof(int);
    
    // Allocate device memory
    cudaError_t error;
    
    error = cudaMalloc(&cuda_data_->d_points_x, float_size);
    if (error != cudaSuccess) goto allocation_error;
    
    error = cudaMalloc(&cuda_data_->d_points_y, float_size);
    if (error != cudaSuccess) goto allocation_error;
    
    error = cudaMalloc(&cuda_data_->d_points_z, float_size);
    if (error != cudaSuccess) goto allocation_error;
    
    error = cudaMalloc(&cuda_data_->d_points_intensity, float_size);
    if (error != cudaSuccess) goto allocation_error;
    
    error = cudaMalloc(&cuda_data_->d_rotation_matrix, 9 * sizeof(float));
    if (error != cudaSuccess) goto allocation_error;
    
    error = cudaMalloc(&cuda_data_->d_translation_vector, 3 * sizeof(float));
    if (error != cudaSuccess) goto allocation_error;
    
    error = cudaMalloc(&cuda_data_->d_output_x, float_size);
    if (error != cudaSuccess) goto allocation_error;
    
    error = cudaMalloc(&cuda_data_->d_output_y, float_size);
    if (error != cudaSuccess) goto allocation_error;
    
    error = cudaMalloc(&cuda_data_->d_output_z, float_size);
    if (error != cudaSuccess) goto allocation_error;
    
    error = cudaMalloc(&cuda_data_->d_output_intensity, float_size);
    if (error != cudaSuccess) goto allocation_error;
    
    error = cudaMalloc(&cuda_data_->d_valid_flags, int_size);
    if (error != cudaSuccess) goto allocation_error;
    
    allocated_points_ = num_points;
    cuda_data_->allocated_size = num_points;
    
    LOG(INFO) << "Allocated GPU memory for " << num_points << " points";
    return true;
    
allocation_error:
    LOG(ERROR) << "Failed to allocate GPU memory: " << cudaGetErrorString(error);
    DeallocateGpuMemory();
    return false;
}

void CudaPointCloudProcessor::DeallocateGpuMemory() {
    if (cuda_data_->d_points_x) { cudaFree(cuda_data_->d_points_x); cuda_data_->d_points_x = nullptr; }
    if (cuda_data_->d_points_y) { cudaFree(cuda_data_->d_points_y); cuda_data_->d_points_y = nullptr; }
    if (cuda_data_->d_points_z) { cudaFree(cuda_data_->d_points_z); cuda_data_->d_points_z = nullptr; }
    if (cuda_data_->d_points_intensity) { cudaFree(cuda_data_->d_points_intensity); cuda_data_->d_points_intensity = nullptr; }
    if (cuda_data_->d_rotation_matrix) { cudaFree(cuda_data_->d_rotation_matrix); cuda_data_->d_rotation_matrix = nullptr; }
    if (cuda_data_->d_translation_vector) { cudaFree(cuda_data_->d_translation_vector); cuda_data_->d_translation_vector = nullptr; }
    if (cuda_data_->d_output_x) { cudaFree(cuda_data_->d_output_x); cuda_data_->d_output_x = nullptr; }
    if (cuda_data_->d_output_y) { cudaFree(cuda_data_->d_output_y); cuda_data_->d_output_y = nullptr; }
    if (cuda_data_->d_output_z) { cudaFree(cuda_data_->d_output_z); cuda_data_->d_output_z = nullptr; }
    if (cuda_data_->d_output_intensity) { cudaFree(cuda_data_->d_output_intensity); cuda_data_->d_output_intensity = nullptr; }
    if (cuda_data_->d_valid_flags) { cudaFree(cuda_data_->d_valid_flags); cuda_data_->d_valid_flags = nullptr; }
    
    allocated_points_ = 0;
    cuda_data_->allocated_size = 0;
}

bool CudaPointCloudProcessor::FilterPointsByDistance(const PointCloudType::Ptr& input_cloud,
                                                    PointCloudType::Ptr& output_cloud,
                                                    float min_distance,
                                                    float max_distance) {
    if (!cuda_initialized_ || !input_cloud || input_cloud->empty()) {
        return false;
    }
    
    const size_t num_points = input_cloud->size();
    
    // Allocate GPU memory if needed
    if (!AllocateGpuMemory(num_points)) {
        return false;
    }
    
    // Prepare host data
    std::vector<float> host_x(num_points), host_y(num_points), host_z(num_points), host_intensity(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        const auto& point = input_cloud->points[i];
        host_x[i] = point.x;
        host_y[i] = point.y;
        host_z[i] = point.z;
        host_intensity[i] = point.intensity;
    }
    
    // Copy data to GPU
    cudaError_t error;
    error = cudaMemcpy(cuda_data_->d_points_x, host_x.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) { LOG(ERROR) << "Failed to copy X data to GPU: " << cudaGetErrorString(error); return false; }
    
    error = cudaMemcpy(cuda_data_->d_points_y, host_y.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) { LOG(ERROR) << "Failed to copy Y data to GPU: " << cudaGetErrorString(error); return false; }
    
    error = cudaMemcpy(cuda_data_->d_points_z, host_z.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) { LOG(ERROR) << "Failed to copy Z data to GPU: " << cudaGetErrorString(error); return false; }
    
    error = cudaMemcpy(cuda_data_->d_points_intensity, host_intensity.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) { LOG(ERROR) << "Failed to copy intensity data to GPU: " << cudaGetErrorString(error); return false; }
    
    // Call CUDA kernel
    error = cuda_impl::cuda_filter_points_by_distance(
        cuda_data_->d_points_x, cuda_data_->d_points_y, cuda_data_->d_points_z, 
        cuda_data_->d_points_intensity, cuda_data_->d_valid_flags,
        min_distance, max_distance, static_cast<int>(num_points)
    );
    
    if (error != cudaSuccess) {
        LOG(ERROR) << "CUDA kernel execution failed: " << cudaGetErrorString(error);
        return false;
    }
    
    // Copy results back
    std::vector<int> valid_flags(num_points);
    error = cudaMemcpy(valid_flags.data(), cuda_data_->d_valid_flags, num_points * sizeof(int), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        LOG(ERROR) << "Failed to copy results from GPU: " << cudaGetErrorString(error);
        return false;
    }
    
    // Filter output cloud
    output_cloud->clear();
    output_cloud->reserve(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        if (valid_flags[i]) {
            output_cloud->push_back(input_cloud->points[i]);
        }
    }
    
    return true;
}

bool CudaPointCloudProcessor::TransformPointCloud(const PointCloudType::Ptr& input_cloud,
                                                 PointCloudType::Ptr& output_cloud,
                                                 const Eigen::Matrix4f& transform) {
    // Extract rotation matrix and translation vector
    float rotation_matrix[9];
    float translation_vector[3];
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            rotation_matrix[i * 3 + j] = transform(i, j);
        }
        translation_vector[i] = transform(i, 3);
    }
    
    return TransformPointCloud(input_cloud, output_cloud, rotation_matrix, translation_vector);
}

bool CudaPointCloudProcessor::TransformPointCloud(const PointCloudType::Ptr& input_cloud,
                                                 PointCloudType::Ptr& output_cloud,
                                                 const float* rotation_matrix,
                                                 const float* translation_vector) {
    if (!cuda_initialized_ || !input_cloud || input_cloud->empty()) {
        return false;
    }
    
    const size_t num_points = input_cloud->size();
    
    // Allocate GPU memory if needed
    if (!AllocateGpuMemory(num_points)) {
        return false;
    }
    
    // Prepare host data
    std::vector<float> host_x(num_points), host_y(num_points), host_z(num_points), host_intensity(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        const auto& point = input_cloud->points[i];
        host_x[i] = point.x;
        host_y[i] = point.y;
        host_z[i] = point.z;
        host_intensity[i] = point.intensity;
    }
    
    // Copy data to GPU
    cudaError_t error;
    error = cudaMemcpy(cuda_data_->d_points_x, host_x.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return false;
    
    error = cudaMemcpy(cuda_data_->d_points_y, host_y.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return false;
    
    error = cudaMemcpy(cuda_data_->d_points_z, host_z.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return false;
    
    error = cudaMemcpy(cuda_data_->d_points_intensity, host_intensity.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return false;
    
    error = cudaMemcpy(cuda_data_->d_rotation_matrix, rotation_matrix, 9 * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return false;
    
    error = cudaMemcpy(cuda_data_->d_translation_vector, translation_vector, 3 * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return false;
    
    // Call CUDA kernel
    error = cuda_impl::cuda_transform_point_cloud(
        cuda_data_->d_points_x, cuda_data_->d_points_y, cuda_data_->d_points_z, cuda_data_->d_points_intensity,
        cuda_data_->d_output_x, cuda_data_->d_output_y, cuda_data_->d_output_z, cuda_data_->d_output_intensity,
        cuda_data_->d_rotation_matrix, cuda_data_->d_translation_vector, static_cast<int>(num_points)
    );
    
    if (error != cudaSuccess) {
        LOG(ERROR) << "CUDA transform kernel execution failed: " << cudaGetErrorString(error);
        return false;
    }
    
    // Copy results back
    std::vector<float> output_x(num_points), output_y(num_points), output_z(num_points), output_intensity(num_points);
    
    error = cudaMemcpy(output_x.data(), cuda_data_->d_output_x, num_points * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return false;
    
    error = cudaMemcpy(output_y.data(), cuda_data_->d_output_y, num_points * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return false;
    
    error = cudaMemcpy(output_z.data(), cuda_data_->d_output_z, num_points * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return false;
    
    error = cudaMemcpy(output_intensity.data(), cuda_data_->d_output_intensity, num_points * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return false;
    
    // Build output cloud
    output_cloud->clear();
    output_cloud->resize(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        auto& point = output_cloud->points[i];
        point.x = output_x[i];
        point.y = output_y[i];
        point.z = output_z[i];
        point.intensity = output_intensity[i];
        // Copy other fields from input
        const auto& input_point = input_cloud->points[i];
        point.normal_x = input_point.normal_x;
        point.normal_y = input_point.normal_y;
        point.normal_z = input_point.normal_z;
        point.curvature = input_point.curvature;
    }
    
    return true;
}

bool CudaPointCloudProcessor::ApplyPGOCorrections(const PointCloudType::Ptr& input_cloud,
                                                PointCloudType::Ptr& output_cloud,
                                                const std::vector<geometry_msgs::PoseStamped>& original_poses,
                                                const std::vector<geometry_msgs::PoseStamped>& corrected_poses,
                                                const std::vector<float>& point_timestamps) {
    if (!cuda_initialized_ || !input_cloud || input_cloud->empty()) {
        return false;
    }

    const size_t num_points = input_cloud->size();
    const size_t num_poses = std::min(original_poses.size(), corrected_poses.size());
    
    if (num_poses == 0) {
        LOG(ERROR) << "No poses provided for PGO correction";
        return false;
    }

    // Allocate GPU memory if needed
    if (!AllocateGpuMemory(num_points)) {
        return false;
    }

    try {
        // Prepare host data for input points
        std::vector<float> host_x(num_points), host_y(num_points), host_z(num_points), host_intensity(num_points);
        std::vector<float> host_timestamps(num_points);
        
        for (size_t i = 0; i < num_points; ++i) {
            const auto& point = input_cloud->points[i];
            host_x[i] = point.x;
            host_y[i] = point.y;
            host_z[i] = point.z;
            host_intensity[i] = point.intensity;
            
            // Use provided timestamps or derive from curvature field
            if (i < point_timestamps.size()) {
                host_timestamps[i] = point_timestamps[i];
            } else {
                host_timestamps[i] = point.curvature;  // Fallback to curvature field
            }
        }

        // Prepare pose data (position + quaternion format: x, y, z, qx, qy, qz, qw)
        std::vector<float> original_pose_data(num_poses * 7);
        std::vector<float> corrected_pose_data(num_poses * 7);
        std::vector<float> pose_timestamps(num_poses);
        
        for (size_t i = 0; i < num_poses; ++i) {
            const auto& orig_pose = original_poses[i];
            const auto& corr_pose = corrected_poses[i];
            
            // Original pose
            size_t base_idx = i * 7;
            original_pose_data[base_idx + 0] = orig_pose.pose.position.x;
            original_pose_data[base_idx + 1] = orig_pose.pose.position.y;
            original_pose_data[base_idx + 2] = orig_pose.pose.position.z;
            original_pose_data[base_idx + 3] = orig_pose.pose.orientation.x;
            original_pose_data[base_idx + 4] = orig_pose.pose.orientation.y;
            original_pose_data[base_idx + 5] = orig_pose.pose.orientation.z;
            original_pose_data[base_idx + 6] = orig_pose.pose.orientation.w;
            
            // Corrected pose
            corrected_pose_data[base_idx + 0] = corr_pose.pose.position.x;
            corrected_pose_data[base_idx + 1] = corr_pose.pose.position.y;
            corrected_pose_data[base_idx + 2] = corr_pose.pose.position.z;
            corrected_pose_data[base_idx + 3] = corr_pose.pose.orientation.x;
            corrected_pose_data[base_idx + 4] = corr_pose.pose.orientation.y;
            corrected_pose_data[base_idx + 5] = corr_pose.pose.orientation.z;
            corrected_pose_data[base_idx + 6] = corr_pose.pose.orientation.w;
            
            // Pose timestamp
            pose_timestamps[i] = orig_pose.header.stamp.toSec();
        }

        // Allocate additional GPU memory for PGO data
        float* d_timestamps = nullptr;
        float* d_original_poses = nullptr;
        float* d_corrected_poses = nullptr;
        float* d_pose_timestamps = nullptr;
        
        cudaError_t error;
        
        // Allocate GPU memory for PGO-specific data
        error = cudaMalloc(&d_timestamps, num_points * sizeof(float));
        if (error != cudaSuccess) {
            LOG(ERROR) << "Failed to allocate timestamps GPU memory: " << cudaGetErrorString(error);
            return false;
        }
        
        error = cudaMalloc(&d_original_poses, num_poses * 7 * sizeof(float));
        if (error != cudaSuccess) {
            cudaFree(d_timestamps);
            LOG(ERROR) << "Failed to allocate original poses GPU memory: " << cudaGetErrorString(error);
            return false;
        }
        
        error = cudaMalloc(&d_corrected_poses, num_poses * 7 * sizeof(float));
        if (error != cudaSuccess) {
            cudaFree(d_timestamps);
            cudaFree(d_original_poses);
            LOG(ERROR) << "Failed to allocate corrected poses GPU memory: " << cudaGetErrorString(error);
            return false;
        }
        
        error = cudaMalloc(&d_pose_timestamps, num_poses * sizeof(float));
        if (error != cudaSuccess) {
            cudaFree(d_timestamps);
            cudaFree(d_original_poses);
            cudaFree(d_corrected_poses);
            LOG(ERROR) << "Failed to allocate pose timestamps GPU memory: " << cudaGetErrorString(error);
            return false;
        }

        // Copy input data to GPU
        error = cudaMemcpy(cuda_data_->d_points_x, host_x.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            LOG(ERROR) << "Failed to copy X data to GPU: " << cudaGetErrorString(error);
            goto cleanup_and_return_false;
        }
        
        error = cudaMemcpy(cuda_data_->d_points_y, host_y.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            LOG(ERROR) << "Failed to copy Y data to GPU: " << cudaGetErrorString(error);
            goto cleanup_and_return_false;
        }
        
        error = cudaMemcpy(cuda_data_->d_points_z, host_z.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            LOG(ERROR) << "Failed to copy Z data to GPU: " << cudaGetErrorString(error);
            goto cleanup_and_return_false;
        }
        
        error = cudaMemcpy(cuda_data_->d_points_intensity, host_intensity.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            LOG(ERROR) << "Failed to copy intensity data to GPU: " << cudaGetErrorString(error);
            goto cleanup_and_return_false;
        }
        
        error = cudaMemcpy(d_timestamps, host_timestamps.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            LOG(ERROR) << "Failed to copy timestamps data to GPU: " << cudaGetErrorString(error);
            goto cleanup_and_return_false;
        }
        
        error = cudaMemcpy(d_original_poses, original_pose_data.data(), num_poses * 7 * sizeof(float), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            LOG(ERROR) << "Failed to copy original poses to GPU: " << cudaGetErrorString(error);
            goto cleanup_and_return_false;
        }
        
        error = cudaMemcpy(d_corrected_poses, corrected_pose_data.data(), num_poses * 7 * sizeof(float), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            LOG(ERROR) << "Failed to copy corrected poses to GPU: " << cudaGetErrorString(error);
            goto cleanup_and_return_false;
        }
        
        error = cudaMemcpy(d_pose_timestamps, pose_timestamps.data(), num_poses * sizeof(float), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            LOG(ERROR) << "Failed to copy pose timestamps to GPU: " << cudaGetErrorString(error);
            goto cleanup_and_return_false;
        }

        // Call CUDA kernel for PGO corrections
        error = cuda_impl::cuda_apply_pgo_corrections(
            cuda_data_->d_points_x, cuda_data_->d_points_y, cuda_data_->d_points_z, cuda_data_->d_points_intensity,
            d_timestamps,
            cuda_data_->d_output_x, cuda_data_->d_output_y, cuda_data_->d_output_z, cuda_data_->d_output_intensity,
            d_original_poses, d_corrected_poses, d_pose_timestamps,
            static_cast<int>(num_poses), static_cast<int>(num_points)
        );
        
        if (error != cudaSuccess) {
            LOG(ERROR) << "CUDA PGO correction kernel execution failed: " << cudaGetErrorString(error);
            goto cleanup_and_return_false;
        }

        // Copy results back
        {
            std::vector<float> output_x(num_points), output_y(num_points), output_z(num_points), output_intensity(num_points);
            
            error = cudaMemcpy(output_x.data(), cuda_data_->d_output_x, num_points * sizeof(float), cudaMemcpyDeviceToHost);
            if (error != cudaSuccess) {
                LOG(ERROR) << "Failed to copy output X from GPU: " << cudaGetErrorString(error);
                goto cleanup_and_return_false;
            }
            
            error = cudaMemcpy(output_y.data(), cuda_data_->d_output_y, num_points * sizeof(float), cudaMemcpyDeviceToHost);
            if (error != cudaSuccess) {
                LOG(ERROR) << "Failed to copy output Y from GPU: " << cudaGetErrorString(error);
                goto cleanup_and_return_false;
            }
            
            error = cudaMemcpy(output_z.data(), cuda_data_->d_output_z, num_points * sizeof(float), cudaMemcpyDeviceToHost);
            if (error != cudaSuccess) {
                LOG(ERROR) << "Failed to copy output Z from GPU: " << cudaGetErrorString(error);
                goto cleanup_and_return_false;
            }
            
            error = cudaMemcpy(output_intensity.data(), cuda_data_->d_output_intensity, num_points * sizeof(float), cudaMemcpyDeviceToHost);
            if (error != cudaSuccess) {
                LOG(ERROR) << "Failed to copy output intensity from GPU: " << cudaGetErrorString(error);
                goto cleanup_and_return_false;
            }

            // Build output cloud
            output_cloud->clear();
            output_cloud->resize(num_points);
            for (size_t i = 0; i < num_points; ++i) {
                auto& point = output_cloud->points[i];
                point.x = output_x[i];
                point.y = output_y[i];
                point.z = output_z[i];
                point.intensity = output_intensity[i];
                
                // Copy other fields from input
                const auto& input_point = input_cloud->points[i];
                point.normal_x = input_point.normal_x;
                point.normal_y = input_point.normal_y;
                point.normal_z = input_point.normal_z;
                point.curvature = input_point.curvature;
            }
        }

        // Cleanup PGO-specific GPU memory
        cudaFree(d_timestamps);
        cudaFree(d_original_poses);
        cudaFree(d_corrected_poses);
        cudaFree(d_pose_timestamps);
        
        LOG(INFO) << "Successfully applied CUDA-accelerated PGO corrections to " << num_points << " points using " << num_poses << " pose pairs";
        return true;

    cleanup_and_return_false:
        // Cleanup on error
        if (d_timestamps) cudaFree(d_timestamps);
        if (d_original_poses) cudaFree(d_original_poses);
        if (d_corrected_poses) cudaFree(d_corrected_poses);
        if (d_pose_timestamps) cudaFree(d_pose_timestamps);
        return false;
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Exception in CUDA PGO corrections: " << e.what();
        return false;
    }
}

bool CudaPointCloudProcessor::DownsamplePointCloud(const PointCloudType::Ptr& input_cloud,
                                                   PointCloudType::Ptr& output_cloud,
                                                   float voxel_size) {
    // For now, fall back to CPU implementation
    LOG(WARNING) << "CUDA downsampling not yet implemented, falling back to CPU";
    return false;
}

void CudaPointCloudProcessor::ClearMemory() {
    if (cuda_initialized_) {
        DeallocateGpuMemory();
    }
}

} // namespace faster_lio

#endif // USE_CUDA

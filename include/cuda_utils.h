#ifndef FASTER_LIO_CUDA_UTILS_H
#define FASTER_LIO_CUDA_UTILS_H

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <memory>
#endif

// Include full PCL headers - safe in header when not compiling CUDA
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <geometry_msgs/PoseStamped.h>
#include <vector>

// Include Jetson optimizations header for complete type definitions
#include "jetson_optimizations.h"

namespace faster_lio {

using PointType = pcl::PointXYZINormal;
using PointCloudType = pcl::PointCloud<PointType>;

#ifdef USE_CUDA
/**
 * CUDA-accelerated point cloud processing class
 * Provides GPU acceleration for point cloud operations in faster-lio
 */
class CudaPointCloudProcessor {
public:
    CudaPointCloudProcessor();
    ~CudaPointCloudProcessor();

    // Static method to check CUDA availability
    static bool IsCudaAvailable();

    // Point cloud filtering by distance
    bool FilterPointsByDistance(const PointCloudType::Ptr& input_cloud,
                               PointCloudType::Ptr& output_cloud,
                               float min_distance,
                               float max_distance);

    // Point cloud transformation using 4x4 matrix
    bool TransformPointCloud(const PointCloudType::Ptr& input_cloud,
                            PointCloudType::Ptr& output_cloud,
                            const Eigen::Matrix4f& transform);

    // Point cloud transformation using rotation matrix and translation vector
    bool TransformPointCloud(const PointCloudType::Ptr& input_cloud,
                            PointCloudType::Ptr& output_cloud,
                            const float* rotation_matrix,
                            const float* translation_vector);

    // Point cloud downsampling
    bool DownsamplePointCloud(const PointCloudType::Ptr& input_cloud,
                             PointCloudType::Ptr& output_cloud,
                             float voxel_size);

    // PGO correction acceleration
    bool ApplyPGOCorrections(const PointCloudType::Ptr& input_cloud,
                           PointCloudType::Ptr& output_cloud,
                           const std::vector<geometry_msgs::PoseStamped>& original_poses,
                           const std::vector<geometry_msgs::PoseStamped>& corrected_poses,
                           const std::vector<float>& point_timestamps);

    // Memory management
    void ClearMemory();

private:
    struct CudaData;
    std::unique_ptr<CudaData> cuda_data_;
    
    bool InitializeCuda();
    void CleanupCuda();
    
    // GPU memory management
    bool AllocateGpuMemory(size_t num_points);
    void DeallocateGpuMemory();
    
    size_t allocated_points_;
    bool cuda_initialized_;
    
    // Jetson platform detection and optimization
    jetson::JetsonInfo jetson_info_;
    jetson::JetsonKernelConfig jetson_config_;
};

// CUDA utility functions (CPU fallbacks when CUDA not available)
namespace cuda_utils {
    bool FilterPointCloudByDistance(const PointCloudType::Ptr& input_cloud,
                                   PointCloudType::Ptr& output_cloud,
                                   float min_distance,
                                   float max_distance);
    
    bool TransformPointCloud(const PointCloudType::Ptr& input_cloud,
                            PointCloudType::Ptr& output_cloud,
                            const Eigen::Matrix4f& transform);
    
    bool DownsamplePointCloud(const PointCloudType::Ptr& input_cloud,
                             PointCloudType::Ptr& output_cloud,
                             float voxel_size);
}

#else // !USE_CUDA

// Stub class when CUDA is not available
class CudaPointCloudProcessor {
public:
    CudaPointCloudProcessor() = default;
    ~CudaPointCloudProcessor() = default;
    
    static bool IsCudaAvailable() { return false; }
    
    bool FilterPointsByDistance(const PointCloudType::Ptr& input_cloud,
                               PointCloudType::Ptr& output_cloud,
                               float min_distance,
                               float max_distance) { return false; }
    
    bool TransformPointCloud(const PointCloudType::Ptr& input_cloud,
                            PointCloudType::Ptr& output_cloud,
                            const Eigen::Matrix4f& transform) { return false; }
                            
    bool TransformPointCloud(const PointCloudType::Ptr& input_cloud,
                            PointCloudType::Ptr& output_cloud,
                            const float* rotation_matrix,
                            const float* translation_vector) { return false; }
    
    bool DownsamplePointCloud(const PointCloudType::Ptr& input_cloud,
                             PointCloudType::Ptr& output_cloud,
                             float voxel_size) { return false; }
    
    void ClearMemory() {}
};

// CPU fallback implementations
namespace cuda_utils {
    bool FilterPointCloudByDistance(const PointCloudType::Ptr& input_cloud,
                                   PointCloudType::Ptr& output_cloud,
                                   float min_distance,
                                   float max_distance);
    
    bool TransformPointCloud(const PointCloudType::Ptr& input_cloud,
                            PointCloudType::Ptr& output_cloud,
                            const Eigen::Matrix4f& transform);
    
    bool DownsamplePointCloud(const PointCloudType::Ptr& input_cloud,
                             PointCloudType::Ptr& output_cloud,
                             float voxel_size);
}

#endif // USE_CUDA

} // namespace faster_lio

#endif // FASTER_LIO_CUDA_UTILS_H

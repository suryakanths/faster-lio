#include "cuda_utils.h"
#include <glog/logging.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

namespace faster_lio {
namespace cuda_utils {

bool FilterPointCloudByDistance(const PointCloudType::Ptr& input_cloud,
                               PointCloudType::Ptr& output_cloud,
                               float min_distance,
                               float max_distance) {
    if (!input_cloud || input_cloud->empty()) {
        return false;
    }
    
    output_cloud->clear();
    output_cloud->reserve(input_cloud->size());
    
    float min_dist_sq = min_distance * min_distance;
    float max_dist_sq = max_distance * max_distance;
    
    for (const auto& point : input_cloud->points) {
        float dist_sq = point.x * point.x + point.y * point.y + point.z * point.z;
        if (dist_sq >= min_dist_sq && dist_sq <= max_dist_sq) {
            output_cloud->push_back(point);
        }
    }
    
    return true;
}

bool TransformPointCloud(const PointCloudType::Ptr& input_cloud,
                        PointCloudType::Ptr& output_cloud,
                        const Eigen::Matrix4f& transform) {
    if (!input_cloud || input_cloud->empty()) {
        return false;
    }
    
    output_cloud->clear();
    output_cloud->resize(input_cloud->size());
    
    for (size_t i = 0; i < input_cloud->size(); ++i) {
        const auto& input_point = input_cloud->points[i];
        auto& output_point = output_cloud->points[i];
        
        Eigen::Vector4f point_homogeneous(input_point.x, input_point.y, input_point.z, 1.0f);
        Eigen::Vector4f transformed_point = transform * point_homogeneous;
        
        output_point.x = transformed_point.x();
        output_point.y = transformed_point.y();
        output_point.z = transformed_point.z();
        output_point.intensity = input_point.intensity;
        output_point.normal_x = input_point.normal_x;
        output_point.normal_y = input_point.normal_y;
        output_point.normal_z = input_point.normal_z;
        output_point.curvature = input_point.curvature;
    }
    
    return true;
}

bool DownsamplePointCloud(const PointCloudType::Ptr& input_cloud,
                         PointCloudType::Ptr& output_cloud,
                         float voxel_size) {
    if (!input_cloud || input_cloud->empty()) {
        return false;
    }
    
    pcl::VoxelGrid<PointType> voxel_filter;
    voxel_filter.setInputCloud(input_cloud);
    voxel_filter.setLeafSize(voxel_size, voxel_size, voxel_size);
    voxel_filter.filter(*output_cloud);
    
    return true;
}

} // namespace cuda_utils
} // namespace faster_lio

#ifndef USE_CUDA
// CPU implementations for when CUDA is not available
bool CudaPointCloudProcessor::DownsamplePointCloud(const PointCloudType::Ptr& input_cloud,
                                                  PointCloudType::Ptr& output_cloud,
                                                  float voxel_size) {
    if (!input_cloud || input_cloud->empty()) {
        return false;
    }
    
    pcl::VoxelGrid<PointType> voxel_filter;
    voxel_filter.setInputCloud(input_cloud);
    voxel_filter.setLeafSize(voxel_size, voxel_size, voxel_size);
    voxel_filter.filter(*output_cloud);
    return true;
}
#endif

// CPU fallback utility functions

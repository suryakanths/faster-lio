#ifndef FASTER_LIO_MAP_COMPRESSION_H
#define FASTER_LIO_MAP_COMPRESSION_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/compression/octree_pointcloud_compression.h>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <memory>

#include "common_lib.h"

namespace faster_lio {

/**
 * @brief Advanced point cloud compression algorithms
 * Provides multiple strategies to reduce point cloud size while preserving structural details
 */
class MapCompression {
public:
    struct CompressionParams {
        float voxel_size;                           // Basic voxel grid size
        float adaptive_voxel_min;                   // Minimum adaptive voxel size
        float adaptive_voxel_max;                   // Maximum adaptive voxel size
        float structure_preservation_threshold;     // Threshold for preserving structural features
        float outlier_std_mul;                      // Standard deviation multiplier for outlier removal
        int outlier_mean_k;                         // K nearest neighbors for outlier removal
        float target_compression_ratio;             // Target compression ratio (0.1-1.0)
        bool preserve_edges;                        // Preserve edge features
        bool preserve_corners;                      // Preserve corner features  
        bool preserve_planar_regions;               // Preserve planar surface details
        bool use_octree_compression;                // Use PCL octree compression
        float octree_resolution;                    // Octree compression resolution
        
        // Constructor with default values
        CompressionParams() :
            voxel_size(0.1f),
            adaptive_voxel_min(0.05f),
            adaptive_voxel_max(0.5f),
            structure_preservation_threshold(0.02f),
            outlier_std_mul(1.0f),
            outlier_mean_k(50),
            target_compression_ratio(0.3f),
            preserve_edges(true),
            preserve_corners(true),
            preserve_planar_regions(true),
            use_octree_compression(false),
            octree_resolution(0.01f) {}
    };

    struct CompressionResult {
        int original_points = 0;
        int compressed_points = 0;
        float compression_ratio = 0.0f;
        double processing_time_ms = 0.0;
        std::string compression_method;
        bool success = false;
    };

    explicit MapCompression(const CompressionParams& params = CompressionParams());
    ~MapCompression() = default;

    /**
     * @brief Compress point cloud using adaptive voxel grid
     * Uses different voxel sizes based on local point density and curvature
     */
    CompressionResult CompressAdaptiveVoxel(const PointCloudType::Ptr& input,
                                          PointCloudType::Ptr& output);

    /**
     * @brief Compress point cloud while preserving structural features
     * Detects and preserves edges, corners, and important geometric features
     */
    CompressionResult CompressStructurePreserving(const PointCloudType::Ptr& input,
                                                 PointCloudType::Ptr& output);

    /**
     * @brief Multi-level compression with LOD (Level of Detail)
     * Creates multiple resolution levels for different viewing distances
     */
    CompressionResult CompressMultiLevel(const PointCloudType::Ptr& input,
                                        std::vector<PointCloudType::Ptr>& output_levels);

    /**
     * @brief Octree-based compression
     * Uses PCL's octree compression for maximum size reduction
     */
    CompressionResult CompressOctree(const PointCloudType::Ptr& input,
                                   std::vector<uint8_t>& compressed_data);

    /**
     * @brief Smart compression that automatically selects best method
     * Analyzes point cloud characteristics and chooses optimal compression
     */
    CompressionResult CompressSmart(const PointCloudType::Ptr& input,
                                  PointCloudType::Ptr& output);

    /**
     * @brief Remove statistical outliers
     */
    bool RemoveOutliers(const PointCloudType::Ptr& input,
                       PointCloudType::Ptr& output);

    /**
     * @brief Detect structural features (edges, corners, planes)
     */
    bool DetectStructuralFeatures(const PointCloudType::Ptr& input,
                                 std::vector<int>& feature_indices);

    /**
     * @brief Calculate local point density
     */
    std::vector<float> CalculateLocalDensity(const PointCloudType::Ptr& input,
                                           float search_radius = 0.1f);

    /**
     * @brief Calculate surface curvature
     */
    std::vector<float> CalculateCurvature(const PointCloudType::Ptr& input,
                                        float search_radius = 0.05f);

    // Getters and setters
    void SetCompressionParams(const CompressionParams& params) { params_ = params; }
    const CompressionParams& GetCompressionParams() const { return params_; }

private:
    CompressionParams params_;
    
    // Internal helper methods
    float CalculateAdaptiveVoxelSize(const PointCloudType::Ptr& input,
                                   const std::vector<float>& densities,
                                   const std::vector<float>& curvatures,
                                   int point_index);
    
    bool IsStructuralFeature(const PointCloudType::Ptr& input,
                           int point_index,
                           const std::vector<float>& curvatures);
    
    void ApplyVoxelGridWithFeaturePreservation(const PointCloudType::Ptr& input,
                                             PointCloudType::Ptr& output,
                                             const std::vector<int>& feature_indices);
};

} // namespace faster_lio

#endif // FASTER_LIO_MAP_COMPRESSION_H

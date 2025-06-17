#include "map_compression.h"
#include <chrono>
#include <algorithm>
#include <glog/logging.h>

namespace faster_lio {

MapCompression::MapCompression(const CompressionParams& params) : params_(params) {
    LOG(INFO) << "MapCompression initialized with target ratio: " << params_.target_compression_ratio;
}

MapCompression::CompressionResult MapCompression::CompressAdaptiveVoxel(
    const PointCloudType::Ptr& input, PointCloudType::Ptr& output) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    CompressionResult result;
    result.original_points = input->size();
    result.compression_method = "AdaptiveVoxel";
    
    if (input->empty()) {
        result.success = false;
        return result;
    }

    try {
        // Calculate local density and curvature
        std::vector<float> densities = CalculateLocalDensity(input, 0.1f);
        std::vector<float> curvatures = CalculateCurvature(input, 0.05f);
        
        // Create adaptive voxel grid
        std::map<std::tuple<int, int, int>, std::vector<int>> voxel_map;
        
        for (size_t i = 0; i < input->size(); ++i) {
            const auto& point = input->points[i];
            
            // Calculate adaptive voxel size based on local characteristics
            float adaptive_size = CalculateAdaptiveVoxelSize(input, densities, curvatures, i);
            
            // Calculate voxel coordinates
            int vx = static_cast<int>(std::floor(point.x / adaptive_size));
            int vy = static_cast<int>(std::floor(point.y / adaptive_size));
            int vz = static_cast<int>(std::floor(point.z / adaptive_size));
            
            auto voxel_key = std::make_tuple(vx, vy, vz);
            voxel_map[voxel_key].push_back(i);
        }
        
        // Select representative points from each voxel
        output->clear();
        output->reserve(voxel_map.size());
        
        for (const auto& voxel : voxel_map) {
            const auto& point_indices = voxel.second;
            
            if (point_indices.size() == 1) {
                output->push_back(input->points[point_indices[0]]);
            } else {
                // Find point with highest curvature (most interesting feature)
                int best_idx = point_indices[0];
                float max_curvature = curvatures[best_idx];
                
                for (int idx : point_indices) {
                    if (curvatures[idx] > max_curvature) {
                        max_curvature = curvatures[idx];
                        best_idx = idx;
                    }
                }
                output->push_back(input->points[best_idx]);
            }
        }
        
        result.compressed_points = output->size();
        result.compression_ratio = static_cast<float>(result.compressed_points) / result.original_points;
        result.success = true;
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "AdaptiveVoxel compression failed: " << e.what();
        result.success = false;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.processing_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return result;
}

MapCompression::CompressionResult MapCompression::CompressStructurePreserving(
    const PointCloudType::Ptr& input, PointCloudType::Ptr& output) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    CompressionResult result;
    result.original_points = input->size();
    result.compression_method = "StructurePreserving";
    
    if (input->empty()) {
        result.success = false;
        return result;
    }

    try {
        // Step 1: Remove outliers
        PointCloudType::Ptr clean_cloud(new PointCloudType);
        if (!RemoveOutliers(input, clean_cloud)) {
            clean_cloud = input;
        }
        
        // Step 2: Detect structural features
        std::vector<int> feature_indices;
        DetectStructuralFeatures(clean_cloud, feature_indices);
        
        // Step 3: Apply voxel grid with feature preservation
        ApplyVoxelGridWithFeaturePreservation(clean_cloud, output, feature_indices);
        
        result.compressed_points = output->size();
        result.compression_ratio = static_cast<float>(result.compressed_points) / result.original_points;
        result.success = true;
        
        LOG(INFO) << "Structure preserving compression: " << result.original_points 
                  << " -> " << result.compressed_points << " points ("
                  << (result.compression_ratio * 100) << "% retained)";
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Structure preserving compression failed: " << e.what();
        result.success = false;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.processing_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return result;
}

MapCompression::CompressionResult MapCompression::CompressSmart(
    const PointCloudType::Ptr& input, PointCloudType::Ptr& output) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    CompressionResult result;
    result.original_points = input->size();
    result.compression_method = "Smart";
    
    if (input->empty()) {
        result.success = false;
        return result;
    }

    try {
        // Analyze point cloud characteristics
        float avg_density = 0.0f;
        std::vector<float> densities = CalculateLocalDensity(input, 0.1f);
        for (float d : densities) {
            avg_density += d;
        }
        avg_density /= densities.size();
        
        std::vector<float> curvatures = CalculateCurvature(input, 0.05f);
        float avg_curvature = 0.0f;
        for (float c : curvatures) {
            avg_curvature += c;
        }
        avg_curvature /= curvatures.size();
        
        // Choose compression strategy based on characteristics
        if (avg_curvature > 0.02f && params_.preserve_edges) {
            // High detail scene - use structure preserving
            result = CompressStructurePreserving(input, output);
            result.compression_method = "Smart-StructurePreserving";
        } else if (avg_density > 1000.0f) {
            // Dense point cloud - use adaptive voxel
            result = CompressAdaptiveVoxel(input, output);
            result.compression_method = "Smart-AdaptiveVoxel";
        } else {
            // Regular scene - use standard voxel grid
            pcl::VoxelGrid<PointType> voxel_filter;
            voxel_filter.setInputCloud(input);
            voxel_filter.setLeafSize(params_.voxel_size, params_.voxel_size, params_.voxel_size);
            voxel_filter.filter(*output);
            
            result.compressed_points = output->size();
            result.compression_ratio = static_cast<float>(result.compressed_points) / result.original_points;
            result.compression_method = "Smart-StandardVoxel";
            result.success = true;
        }
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Smart compression failed: " << e.what();
        result.success = false;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.processing_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return result;
}

bool MapCompression::RemoveOutliers(const PointCloudType::Ptr& input,
                                   PointCloudType::Ptr& output) {
    try {
        pcl::StatisticalOutlierRemoval<PointType> sor;
        sor.setInputCloud(input);
        sor.setMeanK(params_.outlier_mean_k);
        sor.setStddevMulThresh(params_.outlier_std_mul);
        sor.filter(*output);
        return true;
    } catch (const std::exception& e) {
        LOG(ERROR) << "Outlier removal failed: " << e.what();
        return false;
    }
}

bool MapCompression::DetectStructuralFeatures(const PointCloudType::Ptr& input,
                                             std::vector<int>& feature_indices) {
    try {
        feature_indices.clear();
        std::vector<float> curvatures = CalculateCurvature(input, 0.05f);
        
        for (size_t i = 0; i < input->size(); ++i) {
            if (IsStructuralFeature(input, i, curvatures)) {
                feature_indices.push_back(i);
            }
        }
        
        LOG(INFO) << "Detected " << feature_indices.size() << " structural features out of " 
                  << input->size() << " points";
        return true;
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Feature detection failed: " << e.what();
        return false;
    }
}

std::vector<float> MapCompression::CalculateLocalDensity(const PointCloudType::Ptr& input,
                                                       float search_radius) {
    std::vector<float> densities(input->size(), 0.0f);
    
    try {
        pcl::KdTreeFLANN<PointType> kdtree;
        kdtree.setInputCloud(input);
        
        for (size_t i = 0; i < input->size(); ++i) {
            std::vector<int> indices;
            std::vector<float> distances;
            
            int found = kdtree.radiusSearch(input->points[i], search_radius, indices, distances);
            densities[i] = static_cast<float>(found) / (4.0f * M_PI * search_radius * search_radius);
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "Density calculation failed: " << e.what();
    }
    
    return densities;
}

std::vector<float> MapCompression::CalculateCurvature(const PointCloudType::Ptr& input,
                                                    float search_radius) {
    std::vector<float> curvatures(input->size(), 0.0f);
    
    try {
        pcl::NormalEstimation<PointType, pcl::Normal> ne;
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>());
        
        ne.setInputCloud(input);
        ne.setSearchMethod(tree);
        ne.setRadiusSearch(search_radius);
        ne.compute(*normals);
        
        for (size_t i = 0; i < normals->size() && i < input->size(); ++i) {
            curvatures[i] = normals->points[i].curvature;
        }
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Curvature calculation failed: " << e.what();
    }
    
    return curvatures;
}

float MapCompression::CalculateAdaptiveVoxelSize(const PointCloudType::Ptr& input,
                                               const std::vector<float>& densities,
                                               const std::vector<float>& curvatures,
                                               int point_index) {
    if (point_index >= densities.size() || point_index >= curvatures.size()) {
        return params_.voxel_size;
    }
    
    float density = densities[point_index];
    float curvature = curvatures[point_index];
    
    // Higher curvature -> smaller voxels (preserve details)
    // Higher density -> smaller voxels (preserve dense regions)
    float curvature_factor = std::exp(-curvature * 50.0f);  // Exponential decay
    float density_factor = std::exp(-density / 1000.0f);    // Exponential decay
    
    float adaptive_size = params_.voxel_size * curvature_factor * density_factor;
    
    // Clamp to reasonable bounds
    adaptive_size = std::max(adaptive_size, params_.adaptive_voxel_min);
    adaptive_size = std::min(adaptive_size, params_.adaptive_voxel_max);
    
    return adaptive_size;
}

bool MapCompression::IsStructuralFeature(const PointCloudType::Ptr& input,
                                        int point_index,
                                        const std::vector<float>& curvatures) {
    if (point_index >= curvatures.size()) {
        return false;
    }
    
    float curvature = curvatures[point_index];
    
    // High curvature indicates edge or corner
    if (curvature > params_.structure_preservation_threshold) {
        return true;
    }
    
    // TODO: Add more sophisticated feature detection
    // - Edge detection using cross products
    // - Corner detection using eigenvalue analysis
    // - Plane boundary detection
    
    return false;
}

void MapCompression::ApplyVoxelGridWithFeaturePreservation(
    const PointCloudType::Ptr& input,
    PointCloudType::Ptr& output,
    const std::vector<int>& feature_indices) {
    
    // Create set for fast lookup
    std::set<int> feature_set(feature_indices.begin(), feature_indices.end());
    
    // Separate feature and non-feature points
    PointCloudType::Ptr feature_points(new PointCloudType);
    PointCloudType::Ptr regular_points(new PointCloudType);
    
    for (size_t i = 0; i < input->size(); ++i) {
        if (feature_set.count(i)) {
            feature_points->push_back(input->points[i]);
        } else {
            regular_points->push_back(input->points[i]);
        }
    }
    
    // Apply voxel grid to regular points
    PointCloudType::Ptr downsampled_regular(new PointCloudType);
    pcl::VoxelGrid<PointType> voxel_filter;
    voxel_filter.setInputCloud(regular_points);
    voxel_filter.setLeafSize(params_.voxel_size, params_.voxel_size, params_.voxel_size);
    voxel_filter.filter(*downsampled_regular);
    
    // Apply finer voxel grid to feature points to preserve more detail
    PointCloudType::Ptr downsampled_features(new PointCloudType);
    float feature_voxel_size = params_.voxel_size * 0.5f;  // Smaller voxels for features
    voxel_filter.setInputCloud(feature_points);
    voxel_filter.setLeafSize(feature_voxel_size, feature_voxel_size, feature_voxel_size);
    voxel_filter.filter(*downsampled_features);
    
    // Combine results
    output->clear();
    output->reserve(downsampled_regular->size() + downsampled_features->size());
    *output += *downsampled_regular;
    *output += *downsampled_features;
}

} // namespace faster_lio

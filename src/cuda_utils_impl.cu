// CUDA implementation file - Pure CUDA kernels without PCL headers
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <algorithm>

namespace faster_lio {
namespace cuda_impl {

// CUDA kernels for point cloud processing
__global__ void filterPointsByDistanceKernel(const float* points_x,
                                            const float* points_y,
                                            const float* points_z,
                                            const float* points_intensity,
                                            int* valid_flags,
                                            float min_dist_sq,
                                            float max_dist_sq,
                                            int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    float x = points_x[idx];
    float y = points_y[idx];
    float z = points_z[idx];
    float dist_sq = x * x + y * y + z * z;
    
    valid_flags[idx] = (dist_sq >= min_dist_sq && dist_sq <= max_dist_sq) ? 1 : 0;
}

__global__ void transformPointCloudKernel(const float* input_x,
                                         const float* input_y,
                                         const float* input_z,
                                         const float* input_intensity,
                                         float* output_x,
                                         float* output_y,
                                         float* output_z,
                                         float* output_intensity,
                                         const float* rotation_matrix,
                                         const float* translation_vector,
                                         int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    float x = input_x[idx];
    float y = input_y[idx];
    float z = input_z[idx];
    
    // Apply rotation
    output_x[idx] = rotation_matrix[0] * x + rotation_matrix[1] * y + rotation_matrix[2] * z + translation_vector[0];
    output_y[idx] = rotation_matrix[3] * x + rotation_matrix[4] * y + rotation_matrix[5] * z + translation_vector[1];
    output_z[idx] = rotation_matrix[6] * x + rotation_matrix[7] * y + rotation_matrix[8] * z + translation_vector[2];
    output_intensity[idx] = input_intensity[idx];
}

__global__ void downsamplePointCloudKernel(const float* input_x,
                                          const float* input_y,
                                          const float* input_z,
                                          const float* input_intensity,
                                          float* output_x,
                                          float* output_y,
                                          float* output_z,
                                          float* output_intensity,
                                          const int* indices,
                                          int num_output_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_output_points) return;
    
    int input_idx = indices[idx];
    output_x[idx] = input_x[input_idx];
    output_y[idx] = input_y[input_idx];
    output_z[idx] = input_z[input_idx];
    output_intensity[idx] = input_intensity[input_idx];
}

// Host functions that call the kernels
extern "C" {
    cudaError_t cuda_filter_points_by_distance(const float* d_points_x,
                                              const float* d_points_y,
                                              const float* d_points_z,
                                              const float* d_points_intensity,
                                              int* d_valid_flags,
                                              float min_distance,
                                              float max_distance,
                                              int num_points) {
        dim3 blockSize(256);
        dim3 gridSize((num_points + blockSize.x - 1) / blockSize.x);
        
        float min_dist_sq = min_distance * min_distance;
        float max_dist_sq = max_distance * max_distance;
        
        filterPointsByDistanceKernel<<<gridSize, blockSize>>>(
            d_points_x, d_points_y, d_points_z, d_points_intensity,
            d_valid_flags, min_dist_sq, max_dist_sq, num_points
        );
        
        return cudaGetLastError();
    }
    
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
                                          int num_points) {
        dim3 blockSize(256);
        dim3 gridSize((num_points + blockSize.x - 1) / blockSize.x);
        
        transformPointCloudKernel<<<gridSize, blockSize>>>(
            d_input_x, d_input_y, d_input_z, d_input_intensity,
            d_output_x, d_output_y, d_output_z, d_output_intensity,
            d_rotation_matrix, d_translation_vector, num_points
        );
        
        return cudaGetLastError();
    }
    
    cudaError_t cuda_downsample_point_cloud(const float* d_input_x,
                                           const float* d_input_y,
                                           const float* d_input_z,
                                           const float* d_input_intensity,
                                           float* d_output_x,
                                           float* d_output_y,
                                           float* d_output_z,
                                           float* d_output_intensity,
                                           const int* d_indices,
                                           int num_output_points) {
        dim3 blockSize(256);
        dim3 gridSize((num_output_points + blockSize.x - 1) / blockSize.x);
        
        downsamplePointCloudKernel<<<gridSize, blockSize>>>(
            d_input_x, d_input_y, d_input_z, d_input_intensity,
            d_output_x, d_output_y, d_output_z, d_output_intensity,
            d_indices, num_output_points
        );
        
        return cudaGetLastError();
    }
}

} // namespace cuda_impl
} // namespace faster_lio

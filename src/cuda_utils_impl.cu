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

// CUDA kernel for applying PGO pose corrections to point clouds
__global__ void applyPGOCorrectionsKernel(const float* input_x,
                                         const float* input_y,
                                         const float* input_z,
                                         const float* input_intensity,
                                         const float* timestamps,
                                         float* output_x,
                                         float* output_y,
                                         float* output_z,
                                         float* output_intensity,
                                         const float* original_poses,    // 7 floats per pose: x,y,z,qx,qy,qz,qw
                                         const float* corrected_poses,   // 7 floats per pose: x,y,z,qx,qy,qz,qw
                                         const float* pose_timestamps,
                                         int num_poses,
                                         int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    float point_timestamp = timestamps[idx];
    
    // Find the nearest pose timestamp using binary search
    int pose_idx = 0;
    float min_time_diff = fabsf(pose_timestamps[0] - point_timestamp);
    
    for (int i = 1; i < num_poses; i++) {
        float time_diff = fabsf(pose_timestamps[i] - point_timestamp);
        if (time_diff < min_time_diff) {
            min_time_diff = time_diff;
            pose_idx = i;
        }
    }
    
    // Extract original pose
    int orig_base = pose_idx * 7;
    float orig_tx = original_poses[orig_base + 0];
    float orig_ty = original_poses[orig_base + 1];
    float orig_tz = original_poses[orig_base + 2];
    float orig_qx = original_poses[orig_base + 3];
    float orig_qy = original_poses[orig_base + 4];
    float orig_qz = original_poses[orig_base + 5];
    float orig_qw = original_poses[orig_base + 6];
    
    // Extract corrected pose
    int corr_base = pose_idx * 7;
    float corr_tx = corrected_poses[corr_base + 0];
    float corr_ty = corrected_poses[corr_base + 1];
    float corr_tz = corrected_poses[corr_base + 2];
    float corr_qx = corrected_poses[corr_base + 3];
    float corr_qy = corrected_poses[corr_base + 4];
    float corr_qz = corrected_poses[corr_base + 5];
    float corr_qw = corrected_poses[corr_base + 6];
    
    // Get input point
    float px = input_x[idx];
    float py = input_y[idx];
    float pz = input_z[idx];
    
    // Step 1: Transform point from world frame to original pose frame
    // Inverse transform: p_local = R_orig^T * (p_world - t_orig)
    float p_world_x = px - orig_tx;
    float p_world_y = py - orig_ty;
    float p_world_z = pz - orig_tz;
    
    // Convert quaternion to rotation matrix (inverse rotation)
    float q_norm = sqrtf(orig_qx*orig_qx + orig_qy*orig_qy + orig_qz*orig_qz + orig_qw*orig_qw);
    orig_qx /= q_norm; orig_qy /= q_norm; orig_qz /= q_norm; orig_qw /= q_norm;
    
    // Rotation matrix for inverse transform (transpose)
    float r00 = 1 - 2*(orig_qy*orig_qy + orig_qz*orig_qz);
    float r01 = 2*(orig_qx*orig_qy + orig_qw*orig_qz);
    float r02 = 2*(orig_qx*orig_qz - orig_qw*orig_qy);
    float r10 = 2*(orig_qx*orig_qy - orig_qw*orig_qz);
    float r11 = 1 - 2*(orig_qx*orig_qx + orig_qz*orig_qz);
    float r12 = 2*(orig_qy*orig_qz + orig_qw*orig_qx);
    float r20 = 2*(orig_qx*orig_qz + orig_qw*orig_qy);
    float r21 = 2*(orig_qy*orig_qz - orig_qw*orig_qx);
    float r22 = 1 - 2*(orig_qx*orig_qx + orig_qy*orig_qy);
    
    float p_local_x = r00*p_world_x + r10*p_world_y + r20*p_world_z;
    float p_local_y = r01*p_world_x + r11*p_world_y + r21*p_world_z;
    float p_local_z = r02*p_world_x + r12*p_world_y + r22*p_world_z;
    
    // Step 2: Transform point from local frame to corrected world frame
    // Forward transform: p_corrected = R_corr * p_local + t_corr
    q_norm = sqrtf(corr_qx*corr_qx + corr_qy*corr_qy + corr_qz*corr_qz + corr_qw*corr_qw);
    corr_qx /= q_norm; corr_qy /= q_norm; corr_qz /= q_norm; corr_qw /= q_norm;
    
    // Rotation matrix for forward transform
    float c00 = 1 - 2*(corr_qy*corr_qy + corr_qz*corr_qz);
    float c01 = 2*(corr_qx*corr_qy - corr_qw*corr_qz);
    float c02 = 2*(corr_qx*corr_qz + corr_qw*corr_qy);
    float c10 = 2*(corr_qx*corr_qy + corr_qw*corr_qz);
    float c11 = 1 - 2*(corr_qx*corr_qx + corr_qz*corr_qz);
    float c12 = 2*(corr_qy*corr_qz - corr_qw*corr_qx);
    float c20 = 2*(corr_qx*corr_qz - corr_qw*corr_qy);
    float c21 = 2*(corr_qy*corr_qz + corr_qw*corr_qx);
    float c22 = 1 - 2*(corr_qx*corr_qx + corr_qy*corr_qy);
    
    output_x[idx] = c00*p_local_x + c01*p_local_y + c02*p_local_z + corr_tx;
    output_y[idx] = c10*p_local_x + c11*p_local_y + c12*p_local_z + corr_ty;
    output_z[idx] = c20*p_local_x + c21*p_local_y + c22*p_local_z + corr_tz;
    output_intensity[idx] = input_intensity[idx];
}

// Kernel for batch timestamp assignment to points
__global__ void assignTimestampsKernel(float* timestamps,
                                      const int* scan_indices,
                                      const float* scan_timestamps,
                                      int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    timestamps[idx] = scan_timestamps[scan_indices[idx]];
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
                                         int num_points) {
        dim3 blockSize(256);
        dim3 gridSize((num_points + blockSize.x - 1) / blockSize.x);
        
        applyPGOCorrectionsKernel<<<gridSize, blockSize>>>(
            d_input_x, d_input_y, d_input_z, d_input_intensity, d_timestamps,
            d_output_x, d_output_y, d_output_z, d_output_intensity,
            d_original_poses, d_corrected_poses, d_pose_timestamps,
            num_poses, num_points
        );
        
        return cudaGetLastError();
    }
    
    cudaError_t cuda_assign_timestamps(float* d_timestamps,
                                     const int* d_scan_indices,
                                     const float* d_scan_timestamps,
                                     int num_points) {
        dim3 blockSize(256);
        dim3 gridSize((num_points + blockSize.x - 1) / blockSize.x);
        
        assignTimestampsKernel<<<gridSize, blockSize>>>(
            d_timestamps, d_scan_indices, d_scan_timestamps, num_points
        );
        
        return cudaGetLastError();
    }
}

} // namespace cuda_impl
} // namespace faster_lio

#include <ros/ros.h>
#include <faster_lio/SaveOptimizedMap.h>
#include <iostream>

int main(int argc, char** argv) {
    ros::init(argc, argv, "save_optimized_map_client");
    ros::NodeHandle nh;
    
    // Wait for the service to be available
    ROS_INFO("Waiting for SaveOptimizedMap service...");
    ros::service::waitForService("/save_optimized_map");
    ROS_INFO("Service available!");
    
    // Create service client
    ros::ServiceClient client = nh.serviceClient<faster_lio::SaveOptimizedMap>("/save_optimized_map");
    
    // Create service request
    faster_lio::SaveOptimizedMap srv;
    
    // Parse command line arguments
    if (argc > 1) {
        srv.request.file_path = argv[1];
        ROS_INFO("Using custom file path: %s", argv[1]);
    } else {
        srv.request.file_path = "";  // Use default path
        ROS_INFO("Using default file path");
    }
    
    // Configure request parameters
    srv.request.apply_pgo_corrections = true;
    srv.request.use_advanced_compression = true;
    srv.request.voxel_size = 0.1f;  // 10cm voxel size
    srv.request.compression_ratio = 0.3f;  // Target 30% of original points
    srv.request.preserve_structure = true;
    
    ROS_INFO("Service request parameters:");
    ROS_INFO("  Apply PGO corrections: %s", srv.request.apply_pgo_corrections ? "true" : "false");
    ROS_INFO("  Use advanced compression: %s", srv.request.use_advanced_compression ? "true" : "false");
    ROS_INFO("  Voxel size: %.3f", srv.request.voxel_size);
    ROS_INFO("  Target compression ratio: %.2f", srv.request.compression_ratio);
    ROS_INFO("  Preserve structure: %s", srv.request.preserve_structure ? "true" : "false");
    
    // Call the service
    ROS_INFO("Calling SaveOptimizedMap service...");
    
    if (client.call(srv)) {
        if (srv.response.success) {
            ROS_INFO("✓ Map saved successfully!");
            ROS_INFO("  Saved to: %s", srv.response.saved_file_path.c_str());
            ROS_INFO("  Original points: %d", srv.response.original_points);
            ROS_INFO("  Compressed points: %d", srv.response.compressed_points);
            ROS_INFO("  Compression achieved: %.2f%% (%.2f ratio)", 
                     srv.response.compression_achieved * 100.0f,
                     srv.response.compression_achieved);
            ROS_INFO("  File size: %.2f MB", srv.response.file_size_mb);
            
            // Calculate size reduction
            double size_reduction = (1.0 - srv.response.compression_achieved) * 100.0;
            ROS_INFO("  Size reduction: %.1f%%", size_reduction);
            
        } else {
            ROS_ERROR("✗ Failed to save map: %s", srv.response.message.c_str());
            return 1;
        }
    } else {
        ROS_ERROR("Failed to call service");
        return 1;
    }
    
    return 0;
}

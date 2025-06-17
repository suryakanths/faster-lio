#!/usr/bin/env python3

import rospy
from faster_lio.srv import SaveOptimizedMap, SaveOptimizedMapRequest
import sys

def call_save_optimized_map():
    rospy.init_node('save_map_client')
    
    # Wait for the service to be available
    rospy.wait_for_service('/save_optimized_map')
    
    try:
        # Create service proxy
        save_map = rospy.ServiceProxy('/save_optimized_map', SaveOptimizedMap)
        
        # Create request
        req = SaveOptimizedMapRequest()
        
        # Parse command line arguments
        if len(sys.argv) > 1:
            req.file_path = sys.argv[1]
        else:
            req.file_path = ""  # Use default path
            
        req.apply_pgo_corrections = True
        req.use_advanced_compression = True
        req.voxel_size = 0.1  # 10cm voxel size
        req.compression_ratio = 0.3  # Target 30% of original points
        req.preserve_structure = True
        
        rospy.loginfo("Calling SaveOptimizedMap service...")
        rospy.loginfo(f"  File path: {req.file_path or 'default'}")
        rospy.loginfo(f"  Apply PGO corrections: {req.apply_pgo_corrections}")
        rospy.loginfo(f"  Use advanced compression: {req.use_advanced_compression}")
        rospy.loginfo(f"  Voxel size: {req.voxel_size}")
        rospy.loginfo(f"  Target compression ratio: {req.compression_ratio}")
        rospy.loginfo(f"  Preserve structure: {req.preserve_structure}")
        
        # Call service
        response = save_map(req)
        
        # Print results
        if response.success:
            rospy.loginfo("✓ Map saved successfully!")
            rospy.loginfo(f"  Saved to: {response.saved_file_path}")
            rospy.loginfo(f"  Original points: {response.original_points:,}")
            rospy.loginfo(f"  Compressed points: {response.compressed_points:,}")
            rospy.loginfo(f"  Compression achieved: {response.compression_achieved:.2%}")
            rospy.loginfo(f"  File size: {response.file_size_mb:.2f} MB")
        else:
            rospy.logerr(f"✗ Failed to save map: {response.message}")
            
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

if __name__ == '__main__':
    call_save_optimized_map()

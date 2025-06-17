#!/usr/bin/env python3

"""
Test script for the SaveOptimizedMap service functionality
"""

import rospy
from faster_lio.srv import SaveOptimizedMap, SaveOptimizedMapRequest

def test_service():
    """Test the SaveOptimizedMap service"""
    rospy.init_node('test_save_optimized_map')
    
    # Wait for service to be available
    rospy.loginfo("Waiting for SaveOptimizedMap service...")
    try:
        rospy.wait_for_service('/save_optimized_map', timeout=5.0)
        rospy.loginfo("Service found!")
    except rospy.ROSException:
        rospy.logerr("Service not available. Make sure the mapping node is running.")
        return False
    
    # Create service proxy
    save_map_service = rospy.ServiceProxy('/save_optimized_map', SaveOptimizedMap)
    
    # Create request
    req = SaveOptimizedMapRequest()
    req.file_path = "/home/surya/workspaces/slam_ws/test_map.pcd"
    req.apply_pgo_corrections = True
    req.use_advanced_compression = True
    req.voxel_size = 0.1
    req.compression_ratio = 0.5
    req.preserve_structure = True
    
    try:
        rospy.loginfo("Calling service...")
        response = save_map_service(req)
        
        rospy.loginfo(f"Service call result:")
        rospy.loginfo(f"  Success: {response.success}")
        rospy.loginfo(f"  Message: {response.message}")
        rospy.loginfo(f"  Saved file: {response.saved_file_path}")
        rospy.loginfo(f"  Original points: {response.original_points}")
        rospy.loginfo(f"  Compressed points: {response.compressed_points}")
        rospy.loginfo(f"  Compression achieved: {response.compression_achieved}")
        rospy.loginfo(f"  File size: {response.file_size_mb} MB")
        
        return response.success
        
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return False

if __name__ == '__main__':
    test_service()

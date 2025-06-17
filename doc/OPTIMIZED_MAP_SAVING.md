# Optimized Map Saving with PGO Integration

This document describes the advanced map saving functionality in FasterLIO that includes Pose Graph Optimization (PGO) corrections and intelligent compression algorithms.

## Overview

The `SaveOptimizedMap` service provides a comprehensive solution for saving high-quality, optimized point cloud maps with the following features:

- **PGO Integration**: Apply pose corrections from external PGO backends
- **Advanced Compression**: Multiple algorithms to reduce file size while preserving detail
- **Structure Preservation**: Intelligent algorithms that maintain important geometric features
- **Adaptive Processing**: Automatically selects the best compression method based on data characteristics

## Service Interface

### Service Definition
```
# Request
string file_path                 # Path where to save the map (optional, uses default if empty)
bool apply_pgo_corrections      # Whether to apply PGO pose corrections
bool use_advanced_compression   # Use advanced compression algorithms
float32 voxel_size              # Voxel size for downsampling (0 = no downsampling)
float32 compression_ratio       # Target compression ratio (0.1-1.0, smaller = more compression)
bool preserve_structure         # Preserve important structural features during compression
---
# Response
bool success                    # Success flag
string message                  # Status message
string saved_file_path          # Actual file path where map was saved
int32 original_points           # Number of points before compression
int32 compressed_points         # Number of points after compression
float32 compression_achieved    # Actual compression ratio achieved
float64 file_size_mb           # Final file size in MB
```

## Compression Algorithms

### 1. Adaptive Voxel Grid
- **Use Case**: Dense point clouds with varying detail levels
- **Method**: Dynamically adjusts voxel size based on local point density and curvature
- **Benefits**: Preserves fine details in complex areas while simplifying sparse regions

### 2. Structure-Preserving Compression
- **Use Case**: Maps with important geometric features (edges, corners, planes)
- **Method**: Detects structural features and applies different compression levels
- **Benefits**: Maintains map quality for navigation and perception tasks

### 3. Smart Compression
- **Use Case**: General purpose with automatic algorithm selection
- **Method**: Analyzes point cloud characteristics and chooses optimal method
- **Benefits**: No manual tuning required, adapts to different environments

### 4. Multi-Level LOD (Level of Detail)
- **Use Case**: Applications requiring different resolution levels
- **Method**: Creates multiple resolution versions for different viewing distances
- **Benefits**: Efficient rendering and storage for large-scale maps

## PGO Integration

The service integrates with external PGO systems through ROS topics:

### Required Topics
- `/aft_pgo_odom` (nav_msgs/Odometry): Corrected odometry from PGO backend
- `/key_frames_ids` (std_msgs/Header): Keyframe IDs for loop closure events

### Integration Process
1. FasterLIO publishes raw odometry and point clouds
2. External PGO backend processes data and detects loop closures
3. Corrected poses are published back to FasterLIO
4. Service applies corrections when saving optimized maps

## Usage Examples

### Command Line
```bash
# Save with default settings
rosservice call /save_optimized_map

# Save with custom compression
rosservice call /save_optimized_map "{
  file_path: '/path/to/optimized_map.pcd',
  apply_pgo_corrections: true,
  use_advanced_compression: true,
  voxel_size: 0.05,
  compression_ratio: 0.2,
  preserve_structure: true
}"
```

### Python Client
```python
#!/usr/bin/env python3
import rospy
from faster_lio.srv import SaveOptimizedMap, SaveOptimizedMapRequest

rospy.init_node('save_map_client')
rospy.wait_for_service('/save_optimized_map')

save_map = rospy.ServiceProxy('/save_optimized_map', SaveOptimizedMap)

req = SaveOptimizedMapRequest()
req.file_path = "/path/to/optimized_map.pcd"
req.apply_pgo_corrections = True
req.use_advanced_compression = True
req.voxel_size = 0.1
req.compression_ratio = 0.3
req.preserve_structure = True

response = save_map(req)
print(f"Success: {response.success}")
print(f"Saved to: {response.saved_file_path}")
print(f"Compression: {response.original_points} -> {response.compressed_points} points")
```

### Launch File Integration
```xml
<launch>
    <!-- FasterLIO with PCD saving enabled -->
    <node pkg="faster_lio" type="run_mapping_online" name="faster_lio_mapping">
        <param name="pcd_save/pcd_save_en" value="true"/>
        <param name="pcd_save/interval" value="-1"/>
    </node>
    
    <!-- PGO backend (example with SC-LIO-SAM) -->
    <include file="$(find sc_lio_sam)/launch/run_sc_lio_sam.launch">
        <arg name="input_cloud_topic" value="/cloud_registered_lidar"/>
        <arg name="input_odom_topic" value="/Odometry"/>
    </include>
</launch>
```

## Performance Characteristics

### Compression Ratios
- **Basic Voxel Grid**: 10-50% size reduction
- **Adaptive Voxel**: 20-60% size reduction
- **Structure Preserving**: 30-70% size reduction
- **Smart Compression**: Automatically optimized

### Processing Times (typical)
- **100K points**: 50-200ms
- **1M points**: 200-800ms
- **10M points**: 1-5 seconds

### Memory Requirements
- **Peak usage**: ~3x input cloud size
- **CUDA acceleration**: 2x faster with GPU memory

## Configuration

### Launch Parameters
```yaml
# Enable PCD accumulation for global map extraction
pcd_save:
  pcd_save_en: true
  interval: -1  # Save all frames

# Compression settings (optional, can be overridden by service calls)
map_compression:
  default_voxel_size: 0.1
  default_compression_ratio: 0.3
  preserve_structure: true
  enable_cuda_acceleration: true
```

### Runtime Configuration
The compression parameters can be adjusted per service call, allowing dynamic optimization based on use case requirements.

## Troubleshooting

### Common Issues

1. **"No accumulated point cloud available"**
   - Ensure `pcd_save_en: true` in configuration
   - Run mapping for sufficient time to accumulate points

2. **"PGO corrections not available"**
   - Verify PGO backend is running and publishing to correct topics
   - Check topic names and message types

3. **"Compression failed"**
   - Reduce compression ratio (increase value)
   - Disable advanced compression and use basic voxel grid
   - Check available memory

### Performance Optimization

1. **For real-time usage**: Use smaller compression ratios (0.5-0.8)
2. **For storage**: Use aggressive compression (0.1-0.3) with structure preservation
3. **For CUDA systems**: Enable GPU acceleration for faster processing
4. **For Jetson platforms**: Use optimized kernel configurations

## Integration with Existing Workflows

### With SLAM Backends
- Compatible with any PGO system that publishes corrected poses
- Examples: SC-LIO-SAM, LIO-SAM, FAST-LIO2 with external PGO

### With Mapping Pipelines
- Exported maps are standard PCL-compatible PCD files
- Can be used with any PCL-based processing pipeline
- Compatible with ROS navigation stack

### With Visualization Tools
- CloudCompare: Direct PCD loading
- PCL Viewer: Command-line visualization
- RViz: Convert to PointCloud2 messages

This advanced map saving functionality provides a complete solution for creating high-quality, optimized point cloud maps suitable for both research and production applications.

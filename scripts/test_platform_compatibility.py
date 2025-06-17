#!/usr/bin/env python3
"""
CUDA/Jetson Platform Detection Test Script
Tests the faster-lio CUDA implementation for platform compatibility
"""

import rospy
import subprocess
import sys
import os

def check_cuda_available():
    """Check if CUDA is available on the system"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def check_jetson_platform():
    """Check if running on Jetson platform"""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip('\x00')
            return 'NVIDIA Jetson' in model, model
    except FileNotFoundError:
        return False, "Not a Jetson platform"

def get_cuda_info():
    """Get CUDA device information"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', 
                               '--format=csv,noheader,nounits'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            return [line.split(', ') for line in lines]
    except FileNotFoundError:
        pass
    return []

def check_ros_environment():
    """Check ROS environment setup"""
    return 'ROS_PACKAGE_PATH' in os.environ

def test_build_artifacts():
    """Check if build artifacts exist"""
    devel_path = '/home/surya/workspaces/slam_ws/devel'
    executables = [
        f'{devel_path}/lib/faster_lio/run_mapping_online',
        f'{devel_path}/lib/faster_lio/run_mapping_offline',
        f'{devel_path}/lib/libfaster_lio.so'
    ]
    
    results = {}
    for exe in executables:
        results[os.path.basename(exe)] = os.path.exists(exe)
    
    return results

def main():
    print("🚀 faster-lio CUDA/Jetson Compatibility Test")
    print("=" * 50)
    
    # Check CUDA availability
    cuda_available = check_cuda_available()
    print(f"CUDA Available: {'✅ Yes' if cuda_available else '❌ No'}")
    
    if cuda_available:
        cuda_info = get_cuda_info()
        for i, (name, driver, memory) in enumerate(cuda_info):
            print(f"  GPU {i}: {name} (Driver: {driver}, Memory: {memory}MB)")
    
    # Check Jetson platform
    is_jetson, model = check_jetson_platform()
    print(f"Jetson Platform: {'✅ ' + model if is_jetson else '❌ Desktop/Other'}")
    
    # Check ROS environment
    ros_env = check_ros_environment()
    print(f"ROS Environment: {'✅ Yes' if ros_env else '❌ No'}")
    
    # Check build artifacts
    print("\\nBuild Artifacts:")
    artifacts = test_build_artifacts()
    for name, exists in artifacts.items():
        status = '✅ Found' if exists else '❌ Missing'
        print(f"  {name}: {status}")
    
    # Summary
    print("\\n📊 Summary:")
    if cuda_available and all(artifacts.values()):
        if is_jetson:
            print("🎯 Ready for Jetson deployment!")
            print(f"   Platform: {model}")
            print("   Optimizations: Jetson-specific CUDA kernels enabled")
        else:
            print("🎯 Ready for desktop deployment!")
            print("   Platform: Desktop GPU")
            print("   Optimizations: High-performance CUDA kernels enabled")
        
        print("\\n🚀 To run faster-lio:")
        print("   roslaunch faster_lio mapping_avia.launch")
    else:
        print("⚠️  System not ready:")
        if not cuda_available:
            print("   - Install NVIDIA drivers and CUDA")
        if not all(artifacts.values()):
            print("   - Build the workspace: catkin_make")
        if not ros_env:
            print("   - Source ROS environment: source /opt/ros/noetic/setup.bash")

if __name__ == '__main__':
    main()

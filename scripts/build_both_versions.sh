#!/bin/bash

# Script to build both GPU and CPU versions of FasterLIO
# Usage: ./build_both_versions.sh [cpu|gpu|both]

set -e

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BUILD_MODE="${1:-both}"

echo "=== FasterLIO Build Script ==="
echo "Workspace: $WORKSPACE_ROOT"
echo "Build mode: $BUILD_MODE"
echo

cd "$WORKSPACE_ROOT"

build_cpu() {
    echo "=== Building CPU version (no CUDA) ==="
    catkin_make clean
    catkin_make -DUSE_CUDA=OFF
    
    if [ $? -eq 0 ]; then
        echo "✅ CPU build successful!"
        echo "Binary location: $WORKSPACE_ROOT/devel/lib/faster_lio/"
        
        # Create CPU-specific binaries
        cp devel/lib/faster_lio/run_mapping_online devel/lib/faster_lio/run_mapping_online_cpu 2>/dev/null || true
        cp devel/lib/faster_lio/run_mapping_offline devel/lib/faster_lio/run_mapping_offline_cpu 2>/dev/null || true
        cp devel/lib/libfaster_lio.so devel/lib/libfaster_lio_cpu.so 2>/dev/null || true
        echo "CPU binaries saved with '_cpu' suffix"
    else
        echo "❌ CPU build failed!"
        return 1
    fi
}

build_gpu() {
    echo "=== Building GPU version (with CUDA) ==="
    catkin_make clean
    catkin_make -DUSE_CUDA=ON
    
    if [ $? -eq 0 ]; then
        echo "✅ GPU build successful!"
        echo "Binary location: $WORKSPACE_ROOT/devel/lib/faster_lio/"
        
        # Create GPU-specific binaries
        cp devel/lib/faster_lio/run_mapping_online devel/lib/faster_lio/run_mapping_online_gpu 2>/dev/null || true
        cp devel/lib/faster_lio/run_mapping_offline devel/lib/faster_lio/run_mapping_offline_gpu 2>/dev/null || true
        cp devel/lib/libfaster_lio.so devel/lib/libfaster_lio_gpu.so 2>/dev/null || true
        echo "GPU binaries saved with '_gpu' suffix"
    else
        echo "❌ GPU build failed!"
        return 1
    fi
}

check_cuda() {
    if command -v nvcc &> /dev/null; then
        echo "CUDA compiler found: $(nvcc --version | grep "release" | cut -d' ' -f6)"
        return 0
    else
        echo "⚠️  CUDA compiler not found. GPU build may fail."
        return 1
    fi
}

case $BUILD_MODE in
    "cpu")
        build_cpu
        ;;
    "gpu")
        check_cuda
        build_gpu
        ;;
    "both")
        echo "Building both versions..."
        build_cpu
        echo
        if check_cuda; then
            build_gpu
        else
            echo "Skipping GPU build due to missing CUDA"
        fi
        ;;
    *)
        echo "Usage: $0 [cpu|gpu|both]"
        echo "  cpu  - Build CPU-only version"
        echo "  gpu  - Build GPU-accelerated version"
        echo "  both - Build both versions (default)"
        exit 1
        ;;
esac

echo
echo "=== Build Summary ==="
echo "Available binaries in $WORKSPACE_ROOT/devel/lib/faster_lio/:"
ls -la "$WORKSPACE_ROOT/devel/lib/faster_lio/" 2>/dev/null || echo "No binaries found"
echo
echo "Available libraries in $WORKSPACE_ROOT/devel/lib/:"
ls -la "$WORKSPACE_ROOT/devel/lib/libfaster_lio*.so" 2>/dev/null || echo "No libraries found"

echo
echo "=== Usage Instructions ==="
echo "To run the CPU version:"
echo "  rosrun faster_lio run_mapping_online_cpu"
echo "  rosrun faster_lio run_mapping_offline_cpu"
echo
echo "To run the GPU version:"
echo "  rosrun faster_lio run_mapping_online_gpu"
echo "  rosrun faster_lio run_mapping_offline_gpu"
echo
echo "Or use the standard names (will use the last built version):"
echo "  rosrun faster_lio run_mapping_online"
echo "  rosrun faster_lio run_mapping_offline"

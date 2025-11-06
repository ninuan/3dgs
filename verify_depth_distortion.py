#!/usr/bin/env python3
"""
Verification script to check if depth_distortion is actually being computed
"""

import torch
import sys
import os

print("=== Depth Distortion Verification ===\n")

# 1. Check if the rasterizer module can be imported
print("1. Checking rasterizer import...")
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    print("   ✓ GaussianRasterizer imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import rasterizer: {e}")
    sys.exit(1)

# 2. Check if rasterizer source code has depth_distortion
print("\n2. Checking CUDA source files...")
cuda_files = [
    "submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu",
    "submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.h",
    "submodules/diff-gaussian-rasterization/rasterize_points.cu",
]

for filepath in cuda_files:
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            content = f.read()
            if 'depth_distortion' in content or 'depth2' in content:
                print(f"   ✓ {filepath} contains depth distortion code")
            else:
                print(f"   ✗ {filepath} does NOT contain depth distortion code")
    else:
        print(f"   ✗ {filepath} not found")

# 3. Check if gaussian_renderer returns depth_distortion
print("\n3. Checking gaussian_renderer/__init__.py...")
renderer_file = "gaussian_renderer/__init__.py"
if os.path.exists(renderer_file):
    with open(renderer_file, 'r') as f:
        content = f.read()
        if 'depth_distortion' in content:
            print(f"   ✓ gaussian_renderer returns depth_distortion")
            # Count occurrences
            count = content.count('depth_distortion')
            print(f"     Found {count} occurrences of 'depth_distortion'")
        else:
            print(f"   ✗ gaussian_renderer does NOT return depth_distortion")

# 4. Try a minimal render test
print("\n4. Testing minimal rendering with depth_distortion...")
try:
    from scene import GaussianModel
    from gaussian_renderer import render
    from argparse import Namespace

    # Create minimal gaussian model
    gaussians = GaussianModel(sh_degree=0, optimizer_type="default")

    # Create some dummy Gaussians
    num_points = 100
    xyz = torch.randn(num_points, 3).cuda()

    # Initialize minimal model
    gaussians._xyz = xyz
    print(f"   ✓ Created {num_points} dummy Gaussians")

except Exception as e:
    print(f"   ✗ Failed to create minimal setup: {e}")

# 5. Check if rasterizer was recompiled recently
print("\n5. Checking if CUDA extension needs recompilation...")
rasterizer_path = "submodules/diff-gaussian-rasterization"
if os.path.exists(rasterizer_path):
    # Check if build directory exists
    build_dirs = ["build", "dist", "diff_gaussian_rasterization.egg-info"]
    for d in build_dirs:
        full_path = os.path.join(rasterizer_path, d)
        if os.path.exists(full_path):
            print(f"   ✓ Found {d}/ directory")
        else:
            print(f"   ✗ {d}/ directory not found - may need to recompile")

    # Check last modification time
    cu_file = os.path.join(rasterizer_path, "cuda_rasterizer/forward.cu")
    if os.path.exists(cu_file):
        import time
        mtime = os.path.getmtime(cu_file)
        mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        print(f"   Last modified: {mtime_str}")

        # Check if it was modified recently (last 24 hours)
        if time.time() - mtime < 86400:
            print(f"   ⚠ CUDA file was modified recently!")
            print(f"   ⚠ You may need to recompile the extension:")
            print(f"   $ cd {rasterizer_path}")
            print(f"   $ pip install -e .")

print("\n" + "="*50)
print("NEXT STEPS:")
print("="*50)
print("""
If any checks failed above, you need to:

1. Recompile the CUDA extension:
   $ cd submodules/diff-gaussian-rasterization
   $ pip install -e .

2. Verify the compilation succeeded:
   $ python -c "from diff_gaussian_rasterization import GaussianRasterizer; print('OK')"

3. Run training with the debug flag to see depth_distortion values:
   $ python train.py -s <data_path> --iterations 100

4. Check the console output for:
   [Iter XXX] Depth Distortion Stats:
     mean=X.XXXXXX, max=X.XXXXXX

   If mean and max are both 0.000000, the CUDA code is not working.
   If they are > 0, the CUDA code is working but weights may need adjustment.
""")

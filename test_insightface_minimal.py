"""
Minimal test to check if InsightFace model can load without crashing
"""
import os
import os.path as osp
import gc
import sys

print("=" * 60)
print("InsightFace Minimal Test")
print("=" * 60)

try:
    print("\n1. Importing InsightFace...")
    from insightface.model_zoo.arcface_onnx import ArcFaceONNX
    print("   ✓ Import successful")
    
    print("\n2. Checking model file...")
    model_path = osp.join(osp.expanduser('~'), '.insightface', 'models', 'buffalo_sc', 'w600k_mbf.onnx')
    if not osp.exists(model_path):
        print(f"   ✗ Model file not found at: {model_path}")
        sys.exit(1)
    print(f"   ✓ Model file exists: {model_path}")
    
    print("\n3. Loading model (this may take 10-30 seconds)...")
    model = ArcFaceONNX(model_file=model_path)
    print("   ✓ Model loaded")
    
    print("\n4. Preparing model...")
    model.prepare(ctx_id=-1)  # CPU only
    print("   ✓ Model prepared")
    
    print("\n5. Testing embedding generation...")
    import numpy as np
    # Create a dummy 112x112 RGB image
    test_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    embedding = model.get_feat(test_img)
    print(f"   ✓ Generated embedding shape: {embedding.shape}")
    
    print("\n6. Cleaning up...")
    del model
    gc.collect()
    print("   ✓ Cleanup complete")
    
    print("\n" + "=" * 60)
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("=" * 60)
    
except MemoryError as e:
    print(f"\n✗ MEMORY ERROR: {e}")
    print("Your system doesn't have enough RAM to load this model.")
    print("Solutions:")
    print("  1. Close other applications to free up memory")
    print("  2. Use a lighter model (if available)")
    print("  3. Run on a system with more RAM")
    sys.exit(1)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

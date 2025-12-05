#!/usr/bin/env python3
"""
System Readiness Test - Golf Cart Face Recognition
Tests InsightFace backend without loading problematic TensorFlow/DeepFace
"""
import sys
print("="*60)
print(" Golf Cart Face Recognition - System Test")
print("="*60)
print(f"Python: {sys.version}\n")

# Test 1: InsightFace
print("[1/4] Testing InsightFace...")
try:
    import insightface
    from insightface.app import FaceAnalysis
    print("✓ InsightFace imported successfully")
except Exception as e:
    print(f"✗ InsightFace import failed: {e}")
    sys.exit(1)

# Test 2: ONNX Runtime
print("\n[2/4] Testing ONNX Runtime...")
try:
    import onnxruntime
    print(f"✓ ONNX Runtime {onnxruntime.__version__}")
except Exception as e:
    print(f"✗ ONNX Runtime failed: {e}")
    sys.exit(1)

# Test 3: Core dependencies
print("\n[3/4] Testing core dependencies...")
try:
    import cv2
    import numpy as np
    import pymongo
    print(f"✓ OpenCV {cv2.__version__}")
    print(f"✓ NumPy {np.__version__}")
    print(f"✓ PyMongo {pymongo.__version__}")
except Exception as e:
    print(f"✗ Core dependency failed: {e}")
    sys.exit(1)

# Test 4: Initialize Face Recognition (InsightFace only)
print("\n[4/4] Initializing Face Analysis...")
try:
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("✓ Face Analysis initialized (buffalo_l model)")
except Exception as e:
    print(f"✗ Face Analysis initialization failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print(" ✓✓✓ SYSTEM READY - InsightFace Backend Active ✓✓✓")
print("="*60)
print("\nNext steps:")
print("1. Register faces: python register_face.py")
print("2. Run recognition: python recognize_face.py")
print("\nNote: DeepFace backend disabled (TensorFlow conflicts)")
print("      InsightFace provides 95%+ accuracy with ArcFace")

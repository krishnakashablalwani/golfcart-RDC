#!/usr/bin/env python3
"""Test InsightFace without DeepFace/TensorFlow"""
import sys
print(f"Python: {sys.version}\n")

print("Testing InsightFace imports...")
try:
    import insightface
    print("✓ InsightFace imported")
    
    from insightface.app import FaceAnalysis
    print("✓ FaceAnalysis imported")
    
    # Test model loading
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    print("✓ FaceAnalysis initialized")
    
    print("\n✓✓✓ InsightFace is working! ✓✓✓")
    print("\nSuggestion: Use InsightFace-only mode (skip DeepFace/TensorFlow)")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

#!/usr/bin/env python3
"""Test script to verify all imports work correctly"""
import sys
print(f"Python: {sys.version}")

print("Testing imports...")
from recognize_face import FaceRecognitionSystem
print("✓ FaceRecognitionSystem imported")

from register_face import FaceRegistrationSystem
print("✓ FaceRegistrationSystem imported")

print("\n✓✓✓ All imports successful - System is ready! ✓✓✓")

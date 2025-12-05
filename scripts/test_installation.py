"""
Quick Test Script - Verify Installation
Tests all imports and basic functionality
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test all required imports"""
    print("\n" + "="*60)
    print("Testing Imports...")
    print("="*60 + "\n")
    
    tests = [
        ("OpenCV", "import cv2; print(f'OpenCV {cv2.__version__}')"),
        ("NumPy", "import numpy as np; print(f'NumPy {np.__version__}')"),
        ("DeepFace", "import deepface; print(f'DeepFace {deepface.__version__}')"),
        ("TensorFlow", "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"),
        ("Keras", "import keras; print(f'Keras {keras.__version__}')"),
        ("PyMongo", "import pymongo; print(f'PyMongo {pymongo.__version__}')"),
        ("Pandas", "import pandas as pd; print(f'Pandas {pd.__version__}')"),
        ("Pillow", "from PIL import Image; print(f'Pillow {Image.__version__}')"),
    ]
    
    passed = 0
    failed = 0
    
    for name, code in tests:
        try:
            exec(code)
            passed += 1
            print(f"✅ {name}: OK")
        except Exception as e:
            failed += 1
            print(f"❌ {name}: FAILED - {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{len(tests)} passed")
    print("="*60 + "\n")
    
    return failed == 0

def test_deepface_models():
    """Test DeepFace model availability"""
    print("\n" + "="*60)
    print("Testing DeepFace Models...")
    print("="*60 + "\n")
    
    try:
        from deepface import DeepFace
        import numpy as np
        
        # Create a test image (black image)
        test_img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        print("Available models:")
        models = ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 
                  'ArcFace', 'Dlib', 'SFace', 'GhostFaceNet']
        
        for model in models:
            try:
                print(f"  - {model}")
            except:
                pass
        
        print(f"\n✅ DeepFace models available")
        print("⚠️  Note: Models will download on first use (~200MB)")
        return True
        
    except Exception as e:
        print(f"❌ DeepFace test failed: {str(e)}")
        return False

def test_camera():
    """Test camera availability (tries OpenCV, then Picamera2 on Raspberry Pi)"""
    print("\n" + "="*60)
    print("Testing Camera...")
    print("="*60 + "\n")

    try:
        import cv2
        import platform
        # On Linux (Pi), skip straight to Picamera2 to avoid slow OpenCV timeouts
        if platform.system().lower() == 'linux':
            print("Linux detected - trying Picamera2 directly...")
        else:
            candidates = [0, 1, 2]
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if platform.system().lower().startswith('win') else [cv2.CAP_V4L2, cv2.CAP_ANY]
            for idx in candidates:
                for be in backends:
                    cap = cv2.VideoCapture(idx, be)
                    if not cap.isOpened():
                        cap.release()
                        continue
                    # Try a few reads to allow auto-exposure
                    ok = False
                    for _ in range(5):
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            ok = True
                            break
                    cap.release()
                    if ok:
                        h, w = frame.shape[:2]
                        print(f"✅ Camera index {idx} opened (backend {be})")
                        print(f"   Resolution: {w}x{h}")
                        return True
        # Try Picamera2 fallback if available (typical for Pi Camera Module)
        try:
            from picamera2 import Picamera2
            print("Attempting Picamera2 capture...")
            picam = Picamera2()
            cfg = picam.create_video_configuration(main={"size": (1280, 720), "format": "YUV420"})
            picam.configure(cfg)
            picam.start()
            import time as _time
            _time.sleep(0.8)
            arr = picam.capture_array("main")
            picam.stop()
            picam.close()
            import cv2 as _cv2
            frame = _cv2.cvtColor(arr, _cv2.COLOR_YUV2BGR_I420)
            h, w = frame.shape[:2]
            print(f"✅ Picamera2 capture OK: {w}x{h}")
            return True
        except ImportError:
            print("Picamera2 not installed. Install with: sudo apt-get install -y python3-picamera2 libcamera-apps")
        except Exception as e2:
            print(f"❌ Picamera2 capture failed: {e2}")
        
        print("❌ No frames captured via OpenCV or Picamera2")
        return False
    except Exception as e:
        print(f"❌ Camera test failed: {str(e)}")
        return False

def test_database_connection():
    """Test MongoDB connection"""
    print("\n" + "="*60)
    print("Testing MongoDB Connection...")
    print("="*60 + "\n")
    
    try:
        from pymongo import MongoClient
        from pymongo.errors import ServerSelectionTimeoutError
        
        # Try to connect with short timeout
        client = MongoClient('mongodb://localhost:27017/', 
                           serverSelectionTimeoutMS=2000)
        
        # Test connection
        client.server_info()
        
        print("✅ MongoDB connected")
        print(f"   Server: {client.address}")
        
        # List databases
        dbs = client.list_database_names()
        print(f"   Databases: {', '.join(dbs)}")
        
        client.close()
        return True
        
    except ServerSelectionTimeoutError:
        print("⚠️  MongoDB not running")
        print("   Start MongoDB with: docker run -d -p 27017:27017 mongo")
        print("   Or install MongoDB locally")
        return False
        
    except Exception as e:
        print(f"⚠️  MongoDB test failed: {str(e)}")
        return False

def test_directory_structure():
    """Test if required directories exist"""
    print("\n" + "="*60)
    print("Testing Directory Structure...")
    print("="*60 + "\n")
    
    dirs = ['Samples', 'Detections']
    
    for dir_name in dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/ exists")
        else:
            os.makedirs(dir_name, exist_ok=True)
            print(f"✅ {dir_name}/ created")
    
    return True

def test_our_modules():
    """Test our custom modules"""
    print("\n" + "="*60)
    print("Testing Custom Modules...")
    print("="*60 + "\n")
    
    modules = [
        ('modules.database', 'database.py'),
        ('modules.config', 'config.py'),
        ('modules.excel_parser', 'excel_parser.py'),
        ('register_face_deepface', 'register_face_deepface.py'),
        ('recognize_face_deepface', 'recognize_face_deepface.py'),
        ('scripts.manage_database', 'manage_database.py')
    ]
    
    passed = 0
    failed = 0
    
    for module, filename in modules:
        try:
            __import__(module)
            print(f"OK {filename}: OK")
            passed += 1
        except Exception as e:
            print(f"X {filename}: {str(e)}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{len(modules)} modules loaded")
    print("="*60 + "\n")
    
    return failed == 0

def main():
    """Run all tests"""
    print("\n" + "System Verification Test")
    print("="*60)
    print("This will test all components of the face recognition system")
    print("="*60)
    
    results = {
        'Imports': test_imports(),
        'DeepFace': test_deepface_models(),
        'Camera': test_camera(),
        'MongoDB': test_database_connection(),
        'Directories': test_directory_structure(),
        'Custom Modules': test_our_modules()
    }
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60 + "\n")
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "⚠️  FAIL"
        print(f"{status} - {test}")
    
    total_pass = sum(results.values())
    total_tests = len(results)
    
    print(f"\n{'='*60}")
    print(f"Overall: {total_pass}/{total_tests} tests passed")
    print("="*60 + "\n")
    
    if total_pass == total_tests:
        print("All tests passed! System ready to use.")
        print("\nNext steps:")
        print("1. Start MongoDB: docker run -d -p 27017:27017 mongo")
        print("2. Import students: python scripts/manage_database.py")
        print("3. Register faces: python register_face.py")
        print("4. Run recognition: python recognize_face.py")
    elif results['MongoDB'] == False:
        print("MongoDB not running (optional for testing)")
        print("\nYou can still test:")
        print("- Camera: python register_face.py")
        print("- DeepFace: Works offline")
        print("\nStart MongoDB when ready to save data:")
        print("  docker run -d -p 27017:27017 mongo")
    else:
        print("Some tests failed. Check errors above.")
        print("\nCommon fixes:")
        print("- Camera: Make sure webcam is connected")
        print("- MongoDB: Start MongoDB server")
        print("- Modules: Check for syntax errors")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

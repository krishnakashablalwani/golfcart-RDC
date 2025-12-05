"""
Simple camera test to identify which camera index works
"""
import cv2
import sys

print("Testing camera indices...\n")

for idx in range(5):
    print(f"Testing camera index {idx}...")
    
    # Try different backends
    for backend_name, backend_code in [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]:
        try:
            cap = cv2.VideoCapture(idx, backend_code)
            
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Try to read a frame
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    print(f"  ✓ SUCCESS - Camera {idx} works with {backend_name} backend")
                    print(f"    Frame shape: {frame.shape}")
                    cap.release()
                    print(f"\n✅ Use camera index: {idx}")
                    sys.exit(0)
                
                cap.release()
        except Exception as e:
            pass

print("\n❌ No working camera found!")
print("Try:")
print("  1. Check if camera is connected")
print("  2. Check Device Manager (Windows) or 'ls /dev/video*' (Linux)")
print("  3. Try different camera indices: 0, 1, 2, etc.")

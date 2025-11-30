#!/usr/bin/env python3
"""
Standalone camera test: tries OpenCV (V4L2) and then Picamera2 on Raspberry Pi.
Headless by default if no DISPLAY; will print frame shapes.
"""
import time
import sys
import os


def try_opencv(width=1280, height=720, seconds=5, headless=False):
    import cv2
    candidates = [0, 1, 2, 3, 10]
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    for idx in candidates:
        for be in backends:
            cap = cv2.VideoCapture(idx, be)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            ok = False
            for _ in range(10):
                ret, frame = cap.read()
                if ret and frame is not None:
                    ok = True
                    break
                time.sleep(0.05)
            if ok:
                print(f"OpenCV camera opened at index {idx} (backend {be}).")
                start = time.time()
                frames = 0
                while time.time() - start < seconds:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    frames += 1
                    if not headless:
                        cv2.imshow('OpenCV Camera', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                print(f"OpenCV captured {frames} frames; last shape: {None if frame is None else frame.shape}")
                cap.release()
                if not headless:
                    cv2.destroyAllWindows()
                return frames > 0
            cap.release()
    return False


def try_picamera2(width=1280, height=720, seconds=5, headless=False):
    from picamera2 import Picamera2
    import cv2
    picam = Picamera2()
    cfg = picam.create_video_configuration(main={"size": (width, height), "format": "YUV420"})
    picam.configure(cfg)
    picam.start()
    time.sleep(0.5)
    start = time.time()
    print("Picamera2 opened.")
    try:
        frames = 0
        last_shape = None
        while time.time() - start < seconds:
            arr = picam.capture_array("main")
            frame = cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_I420)
            frames += 1
            last_shape = frame.shape
            if not headless:
                cv2.imshow('Picamera2', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        print(f"Picamera2 captured {frames} frames; last shape: {last_shape}")
    finally:
        picam.stop()
        picam.close()
        if not headless:
            cv2.destroyAllWindows()
    return frames > 0


if __name__ == '__main__':
    headless = os.environ.get('DISPLAY') in (None, '', 'unknown')
    # First try OpenCV
    try:
        import cv2  # noqa: F401
        if try_opencv(headless=headless):
            print("✅ OpenCV camera test passed")
            sys.exit(0)
        else:
            print("OpenCV did not capture frames; trying Picamera2...")
    except Exception as e:
        print(f"OpenCV not available or failed: {e}")

    # Then try Picamera2
    try:
        if try_picamera2(headless=headless):
            print("✅ Picamera2 camera test passed")
            sys.exit(0)
    except ImportError:
        print("Picamera2 not installed. Install: sudo apt-get install -y python3-picamera2 libcamera-apps")
    except Exception as e:
        print(f"Picamera2 failed: {e}")

    print("❌ No camera preview available via OpenCV or Picamera2")
    sys.exit(1)

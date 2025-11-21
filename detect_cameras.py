import cv2

def detect_cameras():
    print("Detecting available cameras...")
    available_cameras = []

    for i in range(10):  # Check indices 0 to 9
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"Camera {i}: {width}x{height} @ {fps} FPS")
                available_cameras.append(i)
            cap.release()
        else:
            cap.release()

    if not available_cameras:
        print("No cameras detected.")
    else:
        print(f"Available cameras: {available_cameras}")

if __name__ == "__main__":
    detect_cameras()
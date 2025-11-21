import cv2
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from scipy.spatial.distance import cosine, euclidean, cityblock
from sklearn.metrics.pairwise import cosine_similarity
from threading import Thread, Lock, Event
import time
import os
import urllib.request

# Choice of detector: 'dnn' uses OpenCV DNN (res10 SSD), 'haar' uses Haar cascade fallback
FACE_DETECTOR = 'dnn'
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
PROTO_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt'
MODEL_URL = 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel'


def ensure_dnn_model():
    """Ensure the DNN model files exist locally; download them if missing."""
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)

    proto_path = os.path.join(MODEL_DIR, 'deploy.prototxt')
    model_path = os.path.join(MODEL_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')

    if not os.path.exists(proto_path):
        print('Downloading prototxt...')
        urllib.request.urlretrieve(PROTO_URL, proto_path)
    if not os.path.exists(model_path):
        print('Downloading caffemodel...')
        urllib.request.urlretrieve(MODEL_URL, model_path)

    return proto_path, model_path


def load_dnn_face_detector():
    proto_path, model_path = ensure_dnn_model()
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    return net

CAMERA_INDEX = 2  # 0 for default camera, 1 for external camera, 2 for OBS
MONGODB_URI = "mongodb://localhost:27017/"
DB_NAME = "face_recognition_db"
COLLECTION_NAME = "registered_faces_ultra"
RECOGNITION_THRESHOLD = 0.90  # Threshold for recognition confidence
MIN_FACE_SIZE = 120  # Minimum size of detected face
IMAGE_SIZE = 512  # Size to which face images are resized


def get_database():
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    return collection


def init_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"Camera found at index {CAMERA_INDEX}")
        return cap
    cap.release()

    print(f"Camera {CAMERA_INDEX} not available, scanning...")
    for index in range(5):
        if index == CAMERA_INDEX:
            continue
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print(f"Camera found at index {index}")
            return cap
        cap.release()

    return cv2.VideoCapture(CAMERA_INDEX)


def extract_advanced_features(face_image):
    face_resized = cv2.resize(face_image, (IMAGE_SIZE, IMAGE_SIZE))
    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    features = []

    for size in [(IMAGE_SIZE, IMAGE_SIZE), (IMAGE_SIZE // 2, IMAGE_SIZE // 2)]:
        resized = cv2.resize(gray, size)
        hog = cv2.HOGDescriptor(size, (16, 16), (8, 8), (8, 8), 9)
        hog_feat = hog.compute(resized)
        if hog_feat is not None:
            hog_feat = hog_feat.flatten()
            hog_feat = hog_feat / (np.linalg.norm(hog_feat) + 1e-7)
            features.append(hog_feat[:500])

    def compute_lbp(image, radius, points):
        lbp_image = np.zeros(image.shape, dtype=np.int32)
        for i in range(radius, image.shape[0] - radius):
            for j in range(radius, image.shape[1] - radius):
                center = image[i, j]
                binary = []
                for k in range(points):
                    angle = 2 * np.pi * k / points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    binary.append(1 if image[x, y] >= center else 0)
                lbp_value = sum([b * (2**idx) for idx, b in enumerate(binary)])
                lbp_image[i, j] = lbp_value % 256
        return lbp_image.astype(np.uint8)

    for radius, points in [(1, 8)]:
        lbp = compute_lbp(gray, radius, points)
        hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.append(hist)

    gabor_features = []
    for theta in range(0, 180, 45):
        for sigma in [3, 5]:
            kernel = cv2.getGaborKernel((21, 21), sigma, np.deg2rad(theta), 10, 0.5, 0)
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            gabor_features.append(filtered.mean())
            gabor_features.append(filtered.std())
    features.append(np.array(gabor_features))

    hsv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [64], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    color_hist = np.concatenate(
        [
            cv2.normalize(hist_h, hist_h).flatten(),
            cv2.normalize(hist_s, hist_s).flatten(),
            cv2.normalize(hist_v, hist_v).flatten(),
        ]
    )
    features.append(color_hist)

    grid_features = []
    grid_size = 4
    step_x = IMAGE_SIZE // grid_size
    step_y = IMAGE_SIZE // grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            region = gray[i * step_y : (i + 1) * step_y, j * step_x : (j + 1) * step_x]
            region_hist = cv2.calcHist([region], [0], None, [32], [0, 256])
            grid_features.append(cv2.normalize(region_hist, region_hist).flatten())
    features.append(np.concatenate(grid_features))

    edges = cv2.Canny(gray, 50, 150)
    edge_hist = cv2.calcHist([edges], [0], None, [32], [0, 256])
    features.append(cv2.normalize(edge_hist, edge_hist).flatten())

    combined = np.concatenate(features)
    combined = combined / (np.linalg.norm(combined) + 1e-7)

    return combined


def compare_faces_advanced(encoding1, encoding2):
    enc1 = np.array(encoding1).reshape(1, -1)
    enc2 = np.array(encoding2).reshape(1, -1)

    min_len = min(enc1.shape[1], enc2.shape[1])
    enc1 = enc1[:, :min_len]
    enc2 = enc2[:, :min_len]

    cos_sim = cosine_similarity(enc1, enc2)[0][0]
    cos_sim = (cos_sim + 1) / 2

    correlation = np.corrcoef(enc1.flatten(), enc2.flatten())[0, 1]
    corr_sim = (correlation + 1) / 2

    euc_dist = euclidean(enc1.flatten(), enc2.flatten())
    euc_sim = 1 / (1 + euc_dist)

    man_dist = cityblock(enc1.flatten(), enc2.flatten())
    man_sim = 1 / (1 + man_dist)

    cheb_dist = np.max(np.abs(enc1.flatten() - enc2.flatten()))
    cheb_sim = 1 / (1 + cheb_dist)

    final_similarity = (
        cos_sim * 0.35
        + corr_sim * 0.30
        + euc_sim * 0.20
        + man_sim * 0.10
        + cheb_sim * 0.05
    )

    return final_similarity


def recognize_face(face_image, registered_faces):
    face_encoding = extract_advanced_features(face_image)

    best_match = None
    best_confidence = 0

    for person in registered_faces:
        stored_encoding = person["face_encoding"]
        similarity = compare_faces_advanced(face_encoding, stored_encoding)

        if similarity > best_confidence:
            best_confidence = similarity
            best_match = person

    if best_confidence >= RECOGNITION_THRESHOLD:
        return (
            best_match["roll_number"],
            best_match.get("name", "Unknown"),
            best_confidence,
        )

    return None, None, 0


def run_recognition():
    collection = get_database()

    registered_faces = list(collection.find({}))

    if len(registered_faces) == 0:
        print("⚠ No registered faces found!")
        print("Please run register_ultra.py first.")
        return

    print(f"✓ Loaded {len(registered_faces)} registered faces")

    face_cascade = None
    face_net = None
    if FACE_DETECTOR == 'dnn':
        try:
            face_net = load_dnn_face_detector()
            print('Using OpenCV DNN face detector')
        except Exception as e:
            print('Failed to load DNN detector, falling back to Haar cascade:', e)
            face_net = None

    if face_net is None:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        )
        print('Using Haar cascade face detector')

    camera = init_camera()
    if not camera.isOpened():
        print("Error: Could not open camera")
        return

    print("\n" + "=" * 70)
    print("ULTRA-HIGH ACCURACY FACE RECOGNITION SYSTEM")
    print("Advanced Multi-Metric Comparison")
    print("=" * 70)
    print(f"Threshold: {RECOGNITION_THRESHOLD:.2%} | Min Face: {MIN_FACE_SIZE}px")
    print("\nPress 'q' to quit")
    print("Press 'r' to reload database")
    print("=" * 70 + "\n")

    # Shared state between capture/display thread and worker
    latest_frame = None
    latest_frame_lock = Lock()
    detections = []
    detections_lock = Lock()
    running = Event()
    running.set()

    # Pre-cache stored encodings as numpy arrays for faster comparison
    for p in registered_faces:
        p['face_encoding'] = np.array(p['face_encoding'], dtype=np.float32)

    def worker_loop():
        """Background worker that processes the latest frame when available."""
        nonlocal latest_frame, detections
        last_recognized_local = {}
        while running.is_set():
            frame_to_process = None
            with latest_frame_lock:
                if latest_frame is not None:
                    frame_to_process = latest_frame.copy()
                    latest_frame = None  # drop it (we're processing)
            if frame_to_process is None:
                time.sleep(0.02)
                continue

            try:
                # Run either DNN or Haar face detection
                if face_net is not None:
                    blob = cv2.dnn.blobFromImage(
                        cv2.resize(frame_to_process, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
                    )
                    face_net.setInput(blob)
                    dnn_detections = face_net.forward()
                    faces_local = []
                    h, w = frame_to_process.shape[:2]
                    for i in range(0, dnn_detections.shape[2]):
                        conf = float(dnn_detections[0, 0, i, 2])
                        if conf > 0.5:
                            box = dnn_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (x1, y1, x2, y2) = box.astype('int')
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(w, x2)
                            y2 = min(h, y2)
                            faces_local.append((x1, y1, x2-x1, y2-y1))
                else:
                    gray = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)
                    faces_local = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.05,
                        minNeighbors=7,
                        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )

                new_detections = []
                for (x, y, w, h) in faces_local:
                    face_crop = frame_to_process[y:y+h, x:x+w]
                    # compute encoding (heavy)
                    encoding = extract_advanced_features(face_crop)
                    best_match = None
                    best_conf = 0.0
                    for person in registered_faces:
                        sim = compare_faces_advanced(encoding, person['face_encoding'])
                        if sim > best_conf:
                            best_conf = sim
                            best_match = person

                    if best_conf >= RECOGNITION_THRESHOLD:
                        roll = best_match['roll_number']
                        name = best_match.get('name', 'Unknown')
                        color = (0, 255, 0)
                        # throttle console messages
                        now = datetime.now()
                        if roll not in last_recognized_local or (now - last_recognized_local[roll]).seconds > 5:
                            print(f"✓ Recognized: {roll} - {name} (Confidence: {best_conf:.2%})")
                            last_recognized_local[roll] = now
                    else:
                        roll = None
                        name = None
                        color = (0, 0, 255)

                    new_detections.append({
                        'rect': (x, y, w, h),
                        'roll': roll,
                        'name': name,
                        'conf': best_conf,
                        'color': color
                    })

                with detections_lock:
                    detections.clear()
                    detections.extend(new_detections)

            except Exception as e:
                print(f"Worker error: {e}")
        # worker exiting

    worker = Thread(target=worker_loop, daemon=True)
    worker.start()

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to grab frame")
                break

            # push latest frame for worker (drop if busy)
            with latest_frame_lock:
                latest_frame = frame.copy()

            # draw last known detections quickly
            with detections_lock:
                draw_dets = list(detections)

            for det in draw_dets:
                if not isinstance(det, dict) or 'rect' not in det:
                    continue
                rect = det['rect']
                if not isinstance(rect, (tuple, list)) or len(rect) != 4:
                    continue
                x, y, w, h = rect
                color = det.get('color', (0, 0, 255))
                roll = det.get('roll') or "Unknown"
                name = det.get('name') or "Not Registered"
                conf = det.get('conf', 0.0)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                label_height = 85 if conf else 60
                cv2.rectangle(frame, (x, y-label_height), (x+w, y), color, -1)
                y_text = y - 10
                cv2.putText(frame, f"Roll: {roll}", (x + 5, y_text),
                           cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, name, (x + 5, y_text - 25),
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                if conf:
                    cv2.putText(frame, f"{conf:.1%}", (x + 5, y_text - 50),
                               cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(frame, f"Registered: {len(registered_faces)} | Detections: {len(draw_dets)}",
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Ultra-High Accuracy Mode | Threshold: {RECOGNITION_THRESHOLD:.0%}",
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow('Ultra-Accurate Recognition - Press Q to Quit, R to Reload', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('r'):
                print("Reloading database...")
                registered_faces = list(collection.find({}))
                for p in registered_faces:
                    p['face_encoding'] = np.array(p['face_encoding'], dtype=np.float32)
                print(f"✓ Reloaded {len(registered_faces)} faces")
                with detections_lock:
                    detections.clear()

    finally:
        # stop worker
        running.clear()
        worker.join(timeout=2)

    # Clean up capture resources
    camera.release()
    cv2.destroyAllWindows()
    print("Camera released")


if __name__ == "__main__":
    try:
        run_recognition()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

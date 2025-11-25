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
from dotenv import load_dotenv
from email_sender import send_email
import requests
from timetable_manager import check_class_status

load_dotenv()

# Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "face_recognition_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "registered_faces_ultra")
CAMERA_INDEX_1 = int(os.getenv("CAMERA_INDEX_1", 0))
CAMERA_INDEX_2 = int(os.getenv("CAMERA_INDEX_2", 1))
RECOGNITION_THRESHOLD = 0.90
MIN_FACE_SIZE = 120
IMAGE_SIZE = 256 # Standardized to match registration

# DNN Model Config
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
PROTO_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt'
MODEL_URL = 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel'

def ensure_dnn_model():
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

def get_database():
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    return collection

def init_cameras():
    """Initialize two cameras for stitching."""
    cap1 = cv2.VideoCapture(CAMERA_INDEX_1, cv2.CAP_DSHOW)
    cap2 = cv2.VideoCapture(CAMERA_INDEX_2, cv2.CAP_DSHOW)
    
    caps = [cap1, cap2]
    for cap in caps:
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    return cap1, cap2

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
    color_hist = np.concatenate([
        cv2.normalize(hist_h, hist_h).flatten(),
        cv2.normalize(hist_s, hist_s).flatten(),
        cv2.normalize(hist_v, hist_v).flatten(),
    ])
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
        cos_sim * 0.35 +
        corr_sim * 0.30 +
        euc_sim * 0.20 +
        man_sim * 0.10 +
        cheb_sim * 0.05
    )
    return final_similarity

def run_recognition():
    collection = get_database()
    registered_faces = list(collection.find({}))
    if not registered_faces:
        print("No registered faces found.")
        return

    print(f"Loaded {len(registered_faces)} faces.")

    # --- Vectorization Pre-calculation ---
    face_matrix = None
    face_matrix_centered = None
    face_matrix_centered_norms = None

    if len(registered_faces) > 0:
        # 1. Main Face Matrix (N x D)
        # Ensure encodings are float32
        face_matrix = np.array([p['face_encoding'] for p in registered_faces], dtype=np.float32)
        
        # 2. Centered Matrix for Correlation (N x D)
        # Subtract mean of each row
        face_matrix_centered = face_matrix - face_matrix.mean(axis=1, keepdims=True)
        # Pre-calculate norms for correlation denominator
        face_matrix_centered_norms = np.linalg.norm(face_matrix_centered, axis=1)
    # -------------------------------------
    
    face_net = load_dnn_face_detector()
    cam1, cam2 = init_cameras()
    
    if not cam1.isOpened() and not cam2.isOpened():
        print("Error: Could not open any cameras.")
        return

    latest_frame = None
    latest_frame_lock = Lock()
    detections = []
    detections_lock = Lock()
    running = Event()
    running.set()

    # Removed legacy loop that converted encodings in-place
    # for p in registered_faces:
    #    p['face_encoding'] = np.array(p['face_encoding'], dtype=np.float32)

    def worker_loop():
        nonlocal latest_frame, detections
        last_recognized_local = {}
        while running.is_set():
            frame_to_process = None
            with latest_frame_lock:
                if latest_frame is not None:
                    frame_to_process = latest_frame.copy()
                    latest_frame = None
            if frame_to_process is None:
                time.sleep(0.02)
                continue

            try:
                blob = cv2.dnn.blobFromImage(cv2.resize(frame_to_process, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                face_net.setInput(blob)
                dnn_detections = face_net.forward()
                faces_local = []
                h, w = frame_to_process.shape[:2]
                for i in range(0, dnn_detections.shape[2]):
                    conf = float(dnn_detections[0, 0, i, 2])
                    if conf > 0.5:
                        box = dnn_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x1, y1, x2, y2) = box.astype('int')
                        faces_local.append((max(0, x1), max(0, y1), x2-x1, y2-y1))

                new_detections = []
                for (x, y, w, h) in faces_local:
                    face_crop = frame_to_process[y:y+h, x:x+w]
                    best_match = None
                    best_conf = 0.0
                    
                    if face_matrix is not None:
                        # --- Vectorized Comparison ---
                        # 1. Cosine Similarity (0.35)
                        # Since vectors are normalized, dot product is cosine similarity
                        cos_sims = np.dot(face_matrix, encoding)
                        cos_sims = (cos_sims + 1) / 2  # Normalize to 0-1

                        # 2. Correlation Similarity (0.30)
                        target_centered = encoding - encoding.mean()
                        target_norm = np.linalg.norm(target_centered)
                        # corr = dot(A_centered, B_centered) / (norm(A_centered) * norm(B_centered))
                        corr_sims = np.dot(face_matrix_centered, target_centered) / (face_matrix_centered_norms * target_norm + 1e-7)
                        corr_sims = (corr_sims + 1) / 2 # Normalize to 0-1

                        # 3. Distance Metrics
                        # Calculate difference matrix once: (N x D)
                        diff_matrix = face_matrix - encoding
                        abs_diff_matrix = np.abs(diff_matrix)

                        # Euclidean (0.20)
                        euc_dists = np.linalg.norm(diff_matrix, axis=1)
                        euc_sims = 1 / (1 + euc_dists)

                        # Manhattan (0.10)
                        man_dists = np.sum(abs_diff_matrix, axis=1)
                        man_sims = 1 / (1 + man_dists)

                        # Chebyshev (0.05)
                        cheb_dists = np.max(abs_diff_matrix, axis=1)
                        cheb_sims = 1 / (1 + cheb_dists)

                        # Weighted Sum
                        final_scores = (
                            cos_sims * 0.35 +
                            corr_sims * 0.30 +
                            euc_sims * 0.20 +
                            man_sims * 0.10 +
                            cheb_sims * 0.05
                        )

                        best_idx = np.argmax(final_scores)
                        best_conf = final_scores[best_idx]
                        best_match = registered_faces[best_idx]
                        # -----------------------------

                    if best_conf >= RECOGNITION_THRESHOLD:
                    if best_conf >= RECOGNITION_THRESHOLD:
                        roll = best_match['roll_number']
                        name = best_match.get('name', 'Unknown')
                        color = (0, 255, 0)
                        now = datetime.now()
                        if roll not in last_recognized_local or (now - last_recognized_local[roll]).seconds > 30:
                            print(f"Recognized: {roll} - {name} ({best_conf:.2%})")
                            last_recognized_local[roll] = now
                            
                            # Check Timetable
                            is_busy, subject = check_class_status(roll)
                            if is_busy:
                                print(f"⚠ VIOLATION: {name} should be in {subject}!")
                                # Save image
                                filename = f"{roll}_{now.strftime('%H%M%S')}.jpg"
                                filepath = os.path.join("static", "captures", filename)
                                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                                cv2.imwrite(filepath, frame_to_process)
                                
                                # Report to HOD Server
                                try:
                                    requests.post("http://localhost:5000/report", json={
                                        "roll": roll,
                                        "name": name,
                                        "subject": subject,
                                        "image_file": filename
                                    })
                                    print("Reported to HOD dashboard.")
                                except Exception as e:
                                    print(f"Failed to report violation: {e}")
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
    worker = Thread(target=worker_loop, daemon=True)
    worker.start()

    try:
        while True:
            ret1, frame1 = cam1.read() if cam1.isOpened() else (False, None)
            ret2, frame2 = cam2.read() if cam2.isOpened() else (False, None)
            
            if not ret1 and not ret2:
                print("No frames from any camera.")
                break
                
            # Stitch frames
            if ret1 and ret2:
                # Ensure same height for hstack
                h1, w1 = frame1.shape[:2]
                h2, w2 = frame2.shape[:2]
                
                if h1 != h2:
                    frame2 = cv2.resize(frame2, (int(w2 * h1 / h2), h1))
                
                frame = np.hstack((frame1, frame2))
            elif ret1:
                frame = frame1
            else:
                frame = frame2

            with latest_frame_lock:
                latest_frame = frame.copy()
            
            with detections_lock:
                draw_dets = list(detections)

            for det in draw_dets:
                x, y, w, h = det['rect']
                color = det['color']
                roll = det['roll'] or "Unknown"
                name = det['name'] or "Not Registered"
                conf = det['conf']
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                cv2.putText(frame, f"{name} ({conf:.1%})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow('Recognition (Stitched)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        running.clear()
        worker.join(timeout=2)
        if cam1.isOpened(): cam1.release()
        if cam2.isOpened(): cam2.release()
        cv2.destroyAllWindows()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recognition()

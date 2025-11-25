import cv2
import numpy as np
from datetime import datetime
from pymongo import MongoClient
import base64
import os
import urllib.request
import multiprocessing
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "face_recognition_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "registered_faces_ultra")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))

NUM_SAMPLES = 5
CAPTURE_DELAY_SECONDS = 4
IMAGE_SIZE = 256

ROLL_COLUMN_CANDIDATES = [
    "roll_number",
    "roll number",
    "roll no",
    "roll",
    "rollno",
    "roll_no",
]
NAME_COLUMN_CANDIDATES = ["name", "student name", "full name"]

def standardize_columns(df):
    """Return tuple (roll_col, name_col) after finding acceptable column names."""
    simplemap = {}
    for c in df.columns:
        key = c.strip().lower().replace("_", " ")
        simplemap[key] = c

    def find(col_candidates):
        for cand in col_candidates:
            cand_key = cand.strip().lower().replace("_", " ")
            if cand_key in simplemap:
                return simplemap[cand_key]
        return None

    roll_col = find(ROLL_COLUMN_CANDIDATES)
    name_col = find(NAME_COLUMN_CANDIDATES)
    return roll_col, name_col

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

def init_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap
    cap.release()
    # Fallback search
    for index in range(5):
        if index == CAMERA_INDEX: continue
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
        cap.release()
    return cv2.VideoCapture(CAMERA_INDEX)

def image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buffer).decode("utf-8")

def extract_advanced_features(face_image):
    # ...existing code from register_ultra.py...
    # Re-implementing the feature extraction logic here
    face_resized = cv2.resize(face_image, (IMAGE_SIZE, IMAGE_SIZE))
    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    features = []

    # HOG
    hog_sizes = [(IMAGE_SIZE, IMAGE_SIZE), (IMAGE_SIZE // 2, IMAGE_SIZE // 2)]
    for size in hog_sizes:
        resized = cv2.resize(gray, size)
        hog = cv2.HOGDescriptor(size, (16, 16), (8, 8), (8, 8), 9)
        hog_feat = hog.compute(resized)
        if hog_feat is not None:
            hog_feat = hog_feat.flatten()
            hog_feat = hog_feat / (np.linalg.norm(hog_feat) + 1e-7)
            features.append(hog_feat[:500])

    # LBP
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

    # Gabor
    gabor_features = []
    for theta in range(0, 180, 45):
        for sigma in [3, 5]:
            kernel = cv2.getGaborKernel((21, 21), sigma, np.deg2rad(theta), 10, 0.5, 0)
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            gabor_features.append(filtered.mean())
            gabor_features.append(filtered.std())
    features.append(np.array(gabor_features))

    # Color Hist
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

    # Grid Features
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

    # Edges
    edges = cv2.Canny(gray, 50, 150)
    edge_hist = cv2.calcHist([edges], [0], None, [32], [0, 256])
    features.append(cv2.normalize(edge_hist, edge_hist).flatten())

    combined = np.concatenate(features)
    combined = combined / (np.linalg.norm(combined) + 1e-7)
    return combined.tolist()

def extract_multiple_encodings(face_images):
    with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), len(face_images))) as pool:
        encodings = pool.map(extract_advanced_features, face_images)
    median_encoding = np.median(encodings, axis=0)
    return median_encoding.tolist()

def lookup_name_from_excel(roll, excel_path='students.xlsx'):
    if not os.path.exists(excel_path):
        return None
    try:
        df = pd.read_excel(excel_path)
    except Exception:
        return None
    roll_col, name_col = standardize_columns(df)
    if roll_col is None:
        return None
    matches = df[df[roll_col].astype(str).str.strip() == str(roll).strip()]
    if len(matches) == 0:
        return None
    if name_col is None:
        for c in df.columns:
            if c != roll_col:
                return str(matches.iloc[0][c])
        return None
    return str(matches.iloc[0][name_col])

def register_face():
    collection = get_database()
    face_net = load_dnn_face_detector()
    camera = init_camera()
    
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera initialized. Starting registration loop...")
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Face Detection for UI
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            face_net.setInput(blob)
            detections = face_net.forward()
            faces = []
            h, w = frame.shape[:2]
            for i in range(0, detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype('int')
                    faces.append((x1, y1, x2-x1, y2-y1))

            display_frame = frame.copy()
            for x, y, fw, fh in faces:
                cv2.rectangle(display_frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

            cv2.putText(display_frame, "Press 'C' to Register, 'Q' to Quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Registration", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Registration Flow
                roll_number = input("\nEnter Roll Number: ").strip()
                if not roll_number:
                    print("Roll number required.")
                    continue
                
                # Try to lookup name from Excel
                name = lookup_name_from_excel(roll_number)
                if name:
                    print(f"Found Name: {name}")
                    confirm = input(f"Is this correct? (y/n): ").strip().lower()
                    if confirm != 'y':
                        name = input("Enter Name (optional): ").strip()
                else:
                    name = input("Enter Name (optional): ").strip()

                # Check existing
                if collection.find_one({"roll_number": roll_number}):
                    if input(f"Roll number {roll_number} exists. Overwrite? (y/n): ").lower() != 'y':
                        continue
                    collection.delete_one({"roll_number": roll_number})

                # Capture Samples
                print(f"Capturing {NUM_SAMPLES} samples...")
                samples = []
                while len(samples) < NUM_SAMPLES:
                    ret, frame = camera.read()
                    if not ret: break
                    
                    # Detect again for capture
                    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                    face_net.setInput(blob)
                    detections = face_net.forward()
                    
                    best_face = None
                    max_area = 0
                    h, w = frame.shape[:2]
                    
                    for i in range(0, detections.shape[2]):
                        conf = float(detections[0, 0, i, 2])
                        if conf > 0.5:
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (x1, y1, x2, y2) = box.astype('int')
                            fw, fh = x2-x1, y2-y1
                            if fw*fh > max_area:
                                max_area = fw*fh
                                best_face = (x1, y1, fw, fh)
                    
                    display_frame = frame.copy()
                    if best_face:
                        x, y, fw, fh = best_face
                        cv2.rectangle(display_frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
                        
                        # Auto capture logic could go here, but for now let's just capture if face found
                        # Or use the countdown logic from before. 
                        # For simplicity, I'll just capture 5 frames with a small delay
                        face_img = frame[y:y+fh, x:x+fw]
                        if face_img.size > 0:
                            samples.append(face_img)
                            print(f"Captured sample {len(samples)}/{NUM_SAMPLES}")
                            time.sleep(0.5)
                    
                    cv2.imshow("Registration", display_frame)
                    cv2.waitKey(1)

                if len(samples) == NUM_SAMPLES:
                    print("Processing features...")
                    encoding = extract_multiple_encodings(samples)
                    
                    best_sample = max(samples, key=lambda x: x.shape[0] * x.shape[1])
                    face_b64 = image_to_base64(best_sample)
                    
                    doc = {
                        "roll_number": roll_number,
                        "name": name if name else "Unknown",
                        "face_encoding": encoding,
                        "face_image": face_b64,
                        "registration_date": datetime.now()
                    }
                    collection.insert_one(doc)
                    print("Registration Successful!")
                else:
                    print("Failed to capture enough samples.")

    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    register_face()

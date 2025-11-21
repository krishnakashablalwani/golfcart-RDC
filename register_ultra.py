import cv2
import numpy as np
from datetime import datetime
from pymongo import MongoClient
import base64
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine, euclidean
import os
import urllib.request
import multiprocessing
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
NUM_SAMPLES = 5  # Number of face samples to capture
CAPTURE_DELAY_SECONDS = 2  # Delay between captures in seconds
IMAGE_SIZE = 256  # Size to which face images are resized


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

    cap = cv2.VideoCapture(CAMERA_INDEX)
    return cap


def image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buffer).decode("utf-8")


def extract_advanced_features(face_image):
    face_resized = cv2.resize(face_image, (IMAGE_SIZE, IMAGE_SIZE))

    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    features = []

    hog_sizes = [(IMAGE_SIZE, IMAGE_SIZE), (IMAGE_SIZE // 2, IMAGE_SIZE // 2)]
    for size in hog_sizes:
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

    return combined.tolist()


def extract_multiple_encodings(face_images):
    with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), len(face_images))) as pool:
        encodings = pool.map(extract_advanced_features, face_images)

    median_encoding = np.median(encodings, axis=0)

    return median_encoding.tolist()


def register_face():
    collection = get_database()

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
    print("ULTRA-HIGH ACCURACY FACE REGISTRATION SYSTEM")
    print("Advanced Multi-Feature Extraction")
    print("=" * 70)
    print(f"Samples: {NUM_SAMPLES} | Resolution: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Delay: {CAPTURE_DELAY_SECONDS}s | Camera: {CAMERA_INDEX}")
    print("\nPress 'c' to start registration")
    print("Press 'q' to quit")
    print("=" * 70 + "\n")

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Run either DNN or Haar face detection
            if face_net is not None:
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
                )
                face_net.setInput(blob)
                detections = face_net.forward()
                faces = []
                h, w = frame.shape[:2]
                for i in range(0, detections.shape[2]):
                    conf = float(detections[0, 0, i, 2])
                    if conf > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x1, y1, x2, y2) = box.astype('int')
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(w, x2)
                        y2 = min(h, y2)
                        faces.append((x1, y1, x2-x1, y2-y1))
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=5,
                    minSize=(80, 80),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )

            if len(faces) > 0:
                print(f"Detected {len(faces)} face(s) in frame")

            display_frame = frame.copy()
            for x, y, w, h in faces:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(
                    display_frame,
                    "High Quality Mode",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                display_frame,
                f"Faces: {len(faces)} | Ultra-High Accuracy Mode",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Ultra-Accurate Registration - Press C to Register, Q to Quit",
                display_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Quitting...")
                break
            elif key == ord("c"):
                if len(faces) == 0:
                    print(
                        "✗ No face detected! Ensure good lighting and face the camera."
                    )
                elif len(faces) > 1:
                    print(
                        "✗ Multiple faces detected! Only one person should be in frame."
                    )
                else:

                    roll_number = input("\nEnter Roll Number: ").strip()
                    if not roll_number:
                        print("✗ Roll number cannot be empty!")
                        continue

                    existing = collection.find_one({"roll_number": roll_number})
                    if existing:
                        response = input(
                            f"⚠ Roll number {roll_number} exists. Overwrite? (y/n): "
                        )
                        if response.lower() != "y":
                            print("Registration cancelled.")
                            continue
                        collection.delete_one({"roll_number": roll_number})

                    name = input("Enter Name (optional): ").strip()

                    print(f"\n{'='*70}")
                    print(f"Capturing {NUM_SAMPLES} ultra-high quality samples...")
                    print("Vary your expression slightly between captures")
                    print(f"{'='*70}\n")

                    face_samples = []
                    sample_count = 0
                    countdown = -1
                    countdown_frames = CAPTURE_DELAY_SECONDS * 30

                    while sample_count < NUM_SAMPLES:
                        ret, frame = camera.read()
                        if not ret:
                            break

                        # Run either DNN or Haar face detection
                        if face_net is not None:
                            blob = cv2.dnn.blobFromImage(
                                cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
                            )
                            face_net.setInput(blob)
                            detections = face_net.forward()
                            faces = []
                            h, w = frame.shape[:2]
                            for i in range(0, detections.shape[2]):
                                conf = float(detections[0, 0, i, 2])
                                if conf > 0.5:
                                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                    (x1, y1, x2, y2) = box.astype('int')
                                    x1 = max(0, x1)
                                    y1 = max(0, y1)
                                    x2 = min(w, x2)
                                    y2 = min(h, y2)
                                    faces.append((x1, y1, x2-x1, y2-y1))
                        else:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            gray = cv2.equalizeHist(gray)
                            faces = face_cascade.detectMultiScale(
                                gray,
                                scaleFactor=1.05,
                                minNeighbors=5,
                                minSize=(80, 80),
                                flags=cv2.CASCADE_SCALE_IMAGE,
                            )

                        if len(faces) > 0:
                            print(f"Detected {len(faces)} face(s) during capture")

                        display_frame = frame.copy()

                        if len(faces) > 0:
                            x, y, w, h = faces[0]
                            cv2.rectangle(
                                display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3
                            )

                            if countdown == -1:
                                countdown = countdown_frames
                            elif countdown > 0:
                                seconds_left = int(countdown / 30) + 1
                                cv2.putText(
                                    display_frame,
                                    f"{seconds_left}",
                                    (x + w // 2 - 40, y + h // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    3,
                                    (0, 255, 255),
                                    5,
                                )
                                countdown -= 1
                            elif countdown == 0:

                                face_crop = frame[y : y + h, x : x + w]
                                if face_crop.size > 0 and w > 100 and h > 100:
                                    face_samples.append(face_crop)
                                    sample_count += 1
                                    print(
                                        f"  ✓ Sample {sample_count}/{NUM_SAMPLES} - Size: {w}x{h}"
                                    )

                                    overlay = display_frame.copy()
                                    cv2.rectangle(
                                        overlay,
                                        (0, 0),
                                        (
                                            display_frame.shape[1],
                                            display_frame.shape[0],
                                        ),
                                        (255, 255, 255),
                                        -1,
                                    )
                                    display_frame = cv2.addWeighted(
                                        display_frame, 0.7, overlay, 0.3, 0
                                    )

                                    if sample_count < NUM_SAMPLES:
                                        countdown = countdown_frames
                                else:
                                    print("  ✗ Quality check failed, retrying...")
                                    countdown = countdown_frames

                        cv2.putText(
                            display_frame,
                            f"Sample: {sample_count}/{NUM_SAMPLES}",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2,
                        )

                        if len(faces) == 0:
                            cv2.putText(
                                display_frame,
                                "Keep face in frame",
                                (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2,
                            )

                        cv2.imshow(
                            "Ultra-Accurate Registration - Press C to Register, Q to Quit",
                            display_frame,
                        )

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            face_samples = []
                            break

                    if len(face_samples) != NUM_SAMPLES:
                        print("✗ Registration cancelled - insufficient samples")
                        continue

                    print("\nProcessing features...")
                    start_time = time.time()
                    face_encoding = extract_multiple_encodings(face_samples)
                    end_time = time.time()
                    print(".2f")

                    best_sample = max(
                        face_samples, key=lambda x: x.shape[0] * x.shape[1]
                    )
                    face_image_b64 = image_to_base64(best_sample)

                    document = {
                        "roll_number": roll_number,
                        "name": name if name else "Unknown",
                        "face_encoding": face_encoding,
                        "face_image": face_image_b64,
                        "num_samples": NUM_SAMPLES,
                        "encoding_method": "ultra_advanced_multi_feature",
                        "image_size": IMAGE_SIZE,
                        "registration_date": datetime.now(),
                        "last_updated": datetime.now(),
                    }

                    result = collection.insert_one(document)

                    print(f"\n{'='*70}")
                    print("✓ REGISTRATION SUCCESSFUL!")
                    print(f"  Roll Number: {roll_number}")
                    print(f"  Name: {name if name else 'Unknown'}")
                    print(f"  Samples: {NUM_SAMPLES}")
                    print(f"  Feature Size: {len(face_encoding)}")
                    print(f"  Method: Ultra-Advanced Multi-Feature")
                    print(f"  Database ID: {result.inserted_id}")
                    print(f"  Total Registered: {collection.count_documents({})}")
                    print(f"{'='*70}\n")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released")


if __name__ == "__main__":
    try:
        register_face()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

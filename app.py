"""
Flask web app for face recognition monitoring.
Login with id "mvsr" and password "mvsr_2025".
Real-time detection alerts with violation logging and email notification.
"""

import os
from dotenv import load_dotenv
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from functools import wraps
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import os.path as osp
from typing import Optional, Dict, Tuple

# Load environment variables from .env file
load_dotenv()

cooldown = 20

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.secret_key = "your_secret_key_change_this"  # Change in production

# Login credentials
VALID_ID = "mvsr"
VALID_PASSWORD = "mvsr_2025"

# Email configuration
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "your_email@gmail.com")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "your_app_password")

# Face recognition setup
USE_GPU = os.environ.get("USE_GPU", "1")

try:
    from insightface.model_zoo.arcface_onnx import ArcFaceONNX

    if USE_GPU in ("1", "true", "True"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    model_path = osp.join(osp.expanduser("~"), ".insightface", "models", "buffalo_sc", "w600k_mbf.onnx")
    arcface_model = ArcFaceONNX(model_file=model_path)
    arcface_model.prepare(ctx_id=0 if providers[0] == "CUDAExecutionProvider" else -1)
    logger.info(f"[OK] ArcFace loaded with {providers[0]}")
except Exception as e:
    logger.error(f"Failed to load InsightFace: {e}")
    arcface_model = None

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

from modules.database import db

# Global state for detection
current_detection = {
    "student": None,
    "timestamp": None,
    "embedding_confidence": 0,
    "landmark_confidence": 0,
    "combined_confidence": 0,
}

current_frame = None  # Store the current frame for violation photos
detection_lock = threading.Lock()


class FaceRecognitionMonitor:
    """Real-time face recognition for monitoring"""

    def __init__(self, camera_index: int = 0, width: int = 1920, height: int = 1080):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.known_embeddings = []
        self.known_landmarks = []
        self.known_rolls = []
        self.known_names = []
        self.last_detection = {}
        self.cooldown_seconds = cooldown  # 5 seconds cooldown per student
        self._load_database()
        logger.info(f"Monitor ready: {len(self.known_rolls)} students")

    def _load_database(self):
        try:
            embeddings = db.face_embeddings.find()
            for doc in embeddings:
                roll = doc.get("roll_number")
                emb = doc.get("embedding")
                if roll and emb:
                    student = db.get_student(roll)
                    name = student.get("name", "Unknown") if student else "Unknown"
                    self.known_embeddings.append(np.array(emb, dtype=np.float32))
                    self.known_rolls.append(roll)
                    self.known_names.append(name)
                    self.known_landmarks.append(None)
            logger.info(f"Loaded {len(self.known_embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Database load error: {e}")

    def _get_face_roi_and_landmarks(self, frame: np.ndarray) -> Optional[Tuple]:
        """Detect face and extract landmarks"""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)
        if not results.detections:
            return None

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        x_min = int(bbox.xmin * w)
        y_min = int(bbox.ymin * h)
        x_max = int((bbox.xmin + bbox.width) * w)
        y_max = int((bbox.ymin + bbox.height) * h)

        pad_x = int((x_max - x_min) * 0.2)
        pad_y = int((y_max - y_min) * 0.2)
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(w, x_max + pad_x)
        y_max = min(h, y_max + pad_y)

        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return None

        mesh_results = face_mesh.process(rgb_frame)
        landmarks_3d = None
        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0].landmark
            landmarks_3d = np.array([[l.x, l.y, l.z] for l in landmarks], dtype=np.float32)

        logger.debug(f"Face detected at ({x_min}, {y_min}, {x_max}, {y_max}), landmarks: {landmarks_3d is not None}")
        return roi, landmarks_3d, (x_min, y_min, x_max, y_max)

    def _get_embedding(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        if arcface_model is None:
            return None
        try:
            face_112 = cv2.resize(face_roi, (112, 112))
            embedding = arcface_model.get_feat(face_112)
            emb = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb
        except Exception as e:
            logger.warning(f"Embedding error: {e}")
            return None

    def _landmark_similarity(self, landmarks1: np.ndarray, landmarks2: np.ndarray) -> float:
        if landmarks1 is None or landmarks2 is None:
            return 1.0
        dist = np.linalg.norm(landmarks1 - landmarks2)
        similarity = max(0.0, 1.0 - (dist / 5.0))
        return similarity

    def _match_face(self, embedding: np.ndarray, landmarks: np.ndarray) -> Optional[Dict]:
        if not self.known_embeddings:
            logger.warning("No embeddings loaded in database")
            return None

        embedding_sims = [
            np.dot(embedding, known_emb).item() if isinstance(np.dot(embedding, known_emb), np.ndarray) else float(np.dot(embedding, known_emb))
            for known_emb in self.known_embeddings
        ]

        landmark_sims = [
            self._landmark_similarity(landmarks, known_lm)
            for known_lm in self.known_landmarks
        ]

        has_landmarks = any(lm is not None for lm in self.known_landmarks)
        if has_landmarks:
            combined_sims = [
                0.3 * emb_sim + 0.7 * lm_sim
                for emb_sim, lm_sim in zip(embedding_sims, landmark_sims)
            ]
        else:
            combined_sims = embedding_sims

        max_similarity = max(combined_sims)
        min_idx = combined_sims.index(max_similarity)

        emb_sim = embedding_sims[min_idx]
        lm_sim = landmark_sims[min_idx]

        logger.debug(f"Match attempt - Embedding: {emb_sim:.3f}, Landmark: {lm_sim:.3f}, Combined: {max_similarity:.3f}")

        if max_similarity < 0.55:
            logger.debug(f"Below threshold (0.55): {max_similarity:.3f}")
            return None

        roll = self.known_rolls[min_idx]
        name = self.known_names[min_idx]

        now = datetime.now()
        if roll in self.last_detection:
            if (now - self.last_detection[roll]).total_seconds() < self.cooldown_seconds:
                return None
        self.last_detection[roll] = now

        if landmarks is not None:
            self.known_landmarks[min_idx] = landmarks

        return {
            "roll_number": roll,
            "name": name,
            "embedding_confidence": float(emb_sim),
            "landmark_confidence": float(lm_sim),
            "combined_confidence": float(max_similarity),
        }

    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """Process frame and return detection if any"""
        roi_data = self._get_face_roi_and_landmarks(frame)
        if roi_data:
            roi, landmarks, (x1, y1, x2, y2) = roi_data
            embedding = self._get_embedding(roi)
            if embedding is not None:
                match = self._match_face(embedding, landmarks)
                return match
        return None


# Global monitor instance
monitor = None


def init_monitor():
    global monitor
    if monitor is None:
        monitor = FaceRecognitionMonitor()


# Login decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


# Routes
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_id = request.form.get("user_id", "")
        password = request.form.get("password", "")

        if user_id == VALID_ID and password == VALID_PASSWORD:
            session["user_id"] = user_id
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def dashboard():
    init_monitor()
    return render_template("dashboard.html", total_students=len(monitor.known_rolls))


@app.route("/api/detection")
@login_required
def get_detection():
    """Poll for latest detection"""
    global current_detection
    with detection_lock:
        if current_detection["student"]:
            # Return current detection
            detection = {
                "student": current_detection["student"],
                "roll_number": current_detection.get("roll_number"),
                "name": current_detection["name"],
                "timestamp": current_detection["timestamp"],
                "embedding_confidence": current_detection["embedding_confidence"],
                "landmark_confidence": current_detection["landmark_confidence"],
                "combined_confidence": current_detection["combined_confidence"],
                "email": current_detection.get("email", ""),
            }
            return jsonify(detection)
    return jsonify({"student": None})


@app.route("/api/violation", methods=["POST"])
@login_required
def log_violation():
    """Log violation and send email"""
    global current_detection, current_frame
    data = request.json
    roll_number = data.get("roll_number")
    student_name = data.get("name")
    email = data.get("email", "")
    year = data.get("year", "")
    department_code = data.get("department_code", "")

    logger.info(f"Violation logged for {student_name} ({roll_number})")
    logger.info(f"Email recipient: {email if email else 'NO EMAIL PROVIDED'}")

    # Save violation photo
    if current_frame is not None:
        photo_path = save_violation_photo(current_frame, roll_number, year, department_code)
    else:
        photo_path = None

    # Log to database
    try:
        db.detections.insert_one({
            "roll_number": roll_number,
            "name": student_name,
            "violation": True,
            "timestamp": datetime.now(),
            "location": "Detection Point",
            "photo_path": photo_path,
        })
    except Exception as e:
        logger.error(f"Error logging violation: {e}")

    # Send email if available
    if email:
        logger.info(f"Sending violation email to {email}")
        send_violation_email(student_name, roll_number, email)
    else:
        logger.warning(f"No email found for {student_name}, skipping email notification")

    # Clear detection
    with detection_lock:
        current_detection = {
            "student": None,
            "timestamp": None,
            "embedding_confidence": 0,
            "landmark_confidence": 0,
            "combined_confidence": 0,
        }

    return jsonify({"status": "logged"})


def save_violation_photo(frame: np.ndarray, roll_number: str, year: str, department_code: str) -> str:
    """Save violation photo with folder structure: Violations/[year]/[dept]/[roll]/[timestamp].jpg"""
    try:
        violations_dir = "Violations"
        student_dir = osp.join(violations_dir, year, department_code, roll_number)
        
        # Create directory if it doesn't exist
        os.makedirs(student_dir, exist_ok=True)
        
        # Save photo with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        photo_path = osp.join(student_dir, f"{timestamp}.jpg")
        
        cv2.imwrite(photo_path, frame)
        logger.info(f"Violation photo saved: {photo_path}")
        return photo_path
    except Exception as e:
        logger.error(f"Failed to save violation photo: {e}")
        return None


def send_violation_email(student_name: str, roll_number: str, recipient_email: str):
    """Send violation notification email"""
    try:
        subject = f"Attendance Violation Alert - {student_name}"
        body = f"""
        <html>
            <body>
                <h2>Attendance Violation Detected</h2>
                <p><strong>Student Name:</strong> {student_name}</p>
                <p><strong>Roll Number:</strong> {roll_number}</p>
                <p><strong>Date & Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Location:</strong> Detection Point</p>
                <p>This is an automated notification. Please contact administration for details.</p>
            </body>
        </html>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = SENDER_EMAIL
        msg["To"] = recipient_email

        part = MIMEText(body, "html")
        msg.attach(part)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())

        logger.info(f"Violation email sent to {recipient_email} for {student_name}")
        print(f"✓ Violation email sent to {recipient_email} for {student_name}")
    except Exception as e:
        logger.error(f"Failed to send violation email: {e}")

# Background thread for continuous detection
def detection_loop():
    """Continuous face detection from camera"""
    global current_frame
    init_monitor()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    logger.info("Detection loop started, monitor loaded with embeddings")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Store current frame for violation photos
        current_frame = frame.copy()

        detection = monitor.process_frame(frame)
        if detection:
            logger.info(f"Detection: {detection['name']} - Confidence: {detection['combined_confidence']:.2f}")
            global current_detection
            with detection_lock:
                # Only update if no detection is currently shown
                if not current_detection["student"]:
                    student = db.get_student(detection["roll_number"])
                    current_detection = {
                        "student": detection["name"],
                        "roll_number": detection["roll_number"],
                        "name": detection["name"],
                        "timestamp": datetime.now().isoformat(),
                        "embedding_confidence": detection["embedding_confidence"],
                        "landmark_confidence": detection["landmark_confidence"],
                        "combined_confidence": detection["combined_confidence"],
                        "email": student.get("student_email", "") if student else "",
                        "year": student.get("year", "") if student else "",
                        "department_code": student.get("department_code", "") if student else "",
                    }

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


# Start detection thread on app startup
def startup():
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()


if __name__ == "__main__":
    startup()
    app.run(debug=False, host="0.0.0.0", port=5000)

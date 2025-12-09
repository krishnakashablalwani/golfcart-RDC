"""
3D face recognition: embedding (ArcFace) + MediaPipe 3D landmarks.
This is the prior 3D variant kept separate from the main recognizer.
"""
import os
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
import logging
import os.path as osp
from typing import Optional, Dict, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
    print(f"[OK] ArcFace loaded with {providers[0]}")
except Exception as e:
    logger.error(f"Failed to load InsightFace: {e}")
    arcface_model = None

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

from modules.database import db


class FaceRecognition3D:
    """Face recognition with embedding + 3D landmarks (classic variant)."""

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self.camera_index = camera_index
        self.width = width
        self.height = height

        self.known_embeddings = []
        self.known_landmarks = []
        self.known_rolls = []
        self.known_names = []
        self.last_detection = {}
        self.cooldown_seconds = 0  # cooldown disabled for rapid repeats

        self._load_database()
        logger.info(f"Recognition ready: {len(self.known_rolls)} students + 3D landmarks")

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
        """Detect face with MediaPipe detection, then collect 3D landmarks."""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detection for bounding box
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

        # Landmarks (optional)
        mesh_results = face_mesh.process(rgb_frame)
        landmarks_3d = None
        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0].landmark
            landmarks_3d = np.array([[l.x, l.y, l.z] for l in landmarks], dtype=np.float32)

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
            return 1.0  # fallback to embedding-only if no landmarks yet
        dist = np.linalg.norm(landmarks1 - landmarks2)
        similarity = max(0.0, 1.0 - (dist / 5.0))  # generous scaling to boost scores
        return similarity

    def _match_face(self, embedding: np.ndarray, landmarks: np.ndarray) -> Optional[Dict]:
        if not self.known_embeddings:
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
                0.3 * emb_sim + 0.7 * lm_sim  # landmark-heavy to raise combined
                for emb_sim, lm_sim in zip(embedding_sims, landmark_sims)
            ]
        else:
            combined_sims = embedding_sims

        max_similarity = max(combined_sims)
        min_idx = combined_sims.index(max_similarity)

        emb_sim = embedding_sims[min_idx]
        lm_sim = landmark_sims[min_idx]

        if max_similarity < 0.55:
            return None

        roll = self.known_rolls[min_idx]
        name = self.known_names[min_idx]

        print(f"[MATCH] {name} ({roll})")
        print(f"  Embedding: {emb_sim:.1%} | Landmarks: {lm_sim:.1%} | Combined: {max_similarity:.1%}")

        now = datetime.now()
        if roll in self.last_detection:
            if (now - self.last_detection[roll]).total_seconds() < self.cooldown_seconds:
                logger.info(f"Cooldown active for {roll}")
                return None
        self.last_detection[roll] = now

        if landmarks is not None:
            self.known_landmarks[min_idx] = landmarks

        try:
            db.detections.insert_one({
                "roll_number": roll,
                "name": name,
                "embedding_confidence": float(emb_sim),
                "landmark_confidence": float(lm_sim),
                "combined_confidence": float(max_similarity),
                "timestamp": datetime.now(),
            })
        except Exception:
            pass

        return {
            "roll_number": roll,
            "name": name,
            "embedding_confidence": float(emb_sim),
            "landmark_confidence": float(lm_sim),
            "combined_confidence": float(max_similarity),
        }

    def recognize(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            logger.error("Cannot open camera")
            return

        logger.info("Starting 3D face recognition (Press 'Q' to exit)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()

            roi_data = self._get_face_roi_and_landmarks(frame)
            if roi_data:
                roi, landmarks, (x1, y1, x2, y2) = roi_data
                embedding = self._get_embedding(roi)
                if embedding is not None:
                    match = self._match_face(embedding, landmarks)
                    if match:
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        text = f"{match['name']} ({match['combined_confidence']:.0%})"
                        cv2.putText(display, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(display, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(display, f"Database: {len(self.known_rolls)} students (3D)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("3D Face Recognition", display)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    recognizer = FaceRecognition3D()
    recognizer.recognize()


if __name__ == "__main__":
    main()

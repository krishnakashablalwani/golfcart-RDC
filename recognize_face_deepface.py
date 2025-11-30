"""
High-Resolution Face Recognition System using DeepFace
Real-time face recognition optimized for 5000+ students
Compatible with Python 3.13

Environment tweaks:
- Suppress TensorFlow oneDNN info logs
- Optionally disable oneDNN optimized kernels if desired
"""

import os
# Suppress TF info/warning messages (0 = all logs, 1 = INFO removed, 2 = INFO+WARNING removed, 3 = ERROR only)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# If you want to stop oneDNN optimization message or force legacy behavior, set this to 0
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pickle
from pathlib import Path
import logging
from deepface import DeepFace
import platform

from modules.database import db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FaceRecognitionDeepFace:
    def __init__(self, camera_index: int = 0,
                 high_res_width: int = 1280,
                 high_res_height: int = 720,
                 model_name: str = 'Facenet512',
                 distance_threshold: float = 0.2):
        """
        Initialize face recognition system
        
        Args:
            camera_index: Camera device index
            high_res_width: Camera width
            high_res_height: Camera height
            model_name: DeepFace model ('Facenet512' recommended)
            distance_threshold: Recognition threshold (lower = stricter)
        """
        self.camera_index = camera_index
        self.high_res_width = high_res_width
        self.high_res_height = high_res_height
        self.model_name = model_name
        self.distance_threshold = distance_threshold
        
        # Cache for known faces
        self.known_embeddings = []
        self.known_roll_numbers = []
        self.known_names = []
        
        # Detection cooldown
        self.last_detection = {}
        self.cooldown_seconds = 30
        
        # Cache file
        self.cache_file = Path("face_embeddings_cache.pkl")
        
        # Headless mode detection
        self.headless = os.environ.get('DISPLAY') in (None, '', 'unknown')
        
        logger.info(f"Face Recognition initialized with {model_name}")
        logger.info(f"Distance threshold: {distance_threshold}")
        logger.info(f"Resolution: {high_res_width}x{high_res_height}")
        logger.info(f"Headless mode: {self.headless} (DISPLAY={os.environ.get('DISPLAY', 'NOT SET')})")

    def _open_camera(self):
        """Open camera with platform-specific fallbacks, including Picamera2 on Raspberry Pi."""
        system = platform.system().lower()
        backends = []
        if system.startswith('win'):
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

        for be in backends:
            cap = cv2.VideoCapture(self.camera_index, be)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.high_res_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.high_res_height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                # Validate a frame
                for _ in range(5):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        return cap
                cap.release()

        if system == 'linux':
            try:
                from picamera2 import Picamera2
                class Picam2Capture:
                    def __init__(self, width, height):
                        self.picam = Picamera2()
                        # Try requested size, then fall back to safer modes
                        def _configure(sz):
                            cfg = self.picam.create_video_configuration(
                                main={"size": sz, "format": "YUV420"}
                            )
                            self.picam.configure(cfg)
                        try:
                            _configure((width, height))
                        except Exception as e1:
                            logger.warning(f"Picamera2 {width}x{height} failed: {e1}. Trying 1280x720...")
                            try:
                                _configure((1280, 720))
                            except Exception as e2:
                                logger.warning(f"Picamera2 1280x720 failed: {e2}. Trying 640x480...")
                                try:
                                    _configure((640, 480))
                                except Exception as e3:
                                    logger.warning(f"Picamera2 640x480 failed: {e3}. Trying preview configuration...")
                                    cfg = self.picam.create_preview_configuration(
                                        main={"size": (640, 480), "format": "YUV420"}
                                    )
                                    self.picam.configure(cfg)
                        self.picam.start()
                    def read(self):
                        import cv2 as _cv2
                        arr = self.picam.capture_array("main")
                        frame = _cv2.cvtColor(arr, _cv2.COLOR_YUV2BGR_I420)
                        return True, frame
                    def release(self):
                        try:
                            self.picam.stop()
                            self.picam.close()
                        except Exception:
                            pass
                return Picam2Capture(self.high_res_width, self.high_res_height)
            except Exception as e:
                logger.error(f"Picamera2 fallback failed: {e}")

        return None
    
    def load_known_faces_from_cache(self) -> bool:
        """Load known faces from cache file"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    
                self.known_embeddings = cache_data['embeddings']
                self.known_roll_numbers = cache_data['roll_numbers']
                self.known_names = cache_data['names']
                
                logger.info(f"Loaded {len(self.known_embeddings)} faces from cache")
                
                # Only return True if we actually loaded faces
                return len(self.known_embeddings) > 0
            return False
            
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            return False
    
    def load_known_faces_from_database(self) -> None:
        """Load all registered faces from database"""
        try:
            logger.info("Loading faces from database...")
            
            # Get all face embeddings
            embeddings_data = list(db.face_embeddings.find({'embedding': {'$exists': True}}))
            logger.info(f"Found {len(embeddings_data)} embedding documents in database")
            
            self.known_embeddings = []
            self.known_roll_numbers = []
            self.known_names = []
            
            for data in embeddings_data:
                roll_number = data['roll_number']
                embedding = np.array(data['embedding'])
                
                # Get student info
                student = db.get_student(roll_number)
                if student:
                    self.known_embeddings.append(embedding)
                    self.known_roll_numbers.append(roll_number)
                    self.known_names.append(student.get('name', 'Unknown'))
                    logger.info(f"Loaded: {roll_number} - {student.get('name', 'Unknown')}")
                else:
                    logger.warning(f"Student not found for roll number: {roll_number}")
            
            logger.info(f"Successfully loaded {len(self.known_embeddings)} registered faces")
            
            # Save to cache
            if len(self.known_embeddings) > 0:
                self.save_cache()
            else:
                logger.warning("No embeddings loaded - not saving cache")
            
        except Exception as e:
            logger.error(f"Error loading from database: {str(e)}")
            raise
    
    def save_cache(self) -> None:
        """Save known faces to cache file"""
        try:
            cache_data = {
                'embeddings': self.known_embeddings,
                'roll_numbers': self.known_roll_numbers,
                'names': self.known_names,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Cache saved: {len(self.known_embeddings)} faces")
            
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
    
    def calculate_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine distance between embeddings (for Facenet512)
        
        Returns:
            Distance value (0 = identical, 1 = completely different)
        """
        # Normalize embeddings
        embedding1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-10)
        embedding2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-10)
        
        # Cosine similarity
        cosine_sim = np.dot(embedding1_norm, embedding2_norm)
        
        # Convert to distance (0 = identical, 1 = opposite)
        cosine_distance = 1 - cosine_sim
        
        return float(cosine_distance)
    
    def recognize_face(self, face_embedding: np.ndarray) -> Optional[Dict]:
        """
        Recognize a face from its embedding
        
        Returns:
            Dictionary with student info or None if not recognized
        """
        try:
            if len(self.known_embeddings) == 0:
                return None
            
            # Calculate distances to all known faces
            distances = [
                self.calculate_distance(face_embedding, known_emb)
                for known_emb in self.known_embeddings
            ]
            
            # Find minimum distance
            min_distance = min(distances)
            min_index = distances.index(min_distance)
            
            # Calculate confidence based on distance ratio
            # Facenet512 uses cosine distance (0 = identical, 1 = completely different)
            # Excellent match: < 0.3, Good: 0.3-0.5, Fair: 0.5-0.7
            if len(distances) == 1:
                # Single person: confidence based on cosine distance
                if min_distance < 0.2:
                    confidence = 0.99 - (min_distance * 2)  # 99% at 0, 95% at 0.2
                elif min_distance < 0.4:
                    confidence = 0.95 - ((min_distance - 0.2) * 2.5)  # 95% at 0.2, 85% at 0.4
                elif min_distance < 0.6:
                    confidence = 0.85 - ((min_distance - 0.4) * 2.5)  # 85% at 0.4, 70% at 0.6
                else:
                    confidence = max(0.1, 0.7 - ((min_distance - 0.6) * 1.5))  # Gradually decrease
            else:
                # Multiple people: confidence based on how much better the best match is vs second best
                sorted_distances = sorted(distances)
                second_min = sorted_distances[1] if len(sorted_distances) > 1 else min_distance * 2
                
                # Calculate relative confidence
                if second_min > 0:
                    confidence = 1 - (min_distance / second_min)
                else:
                    confidence = 0.5
                
                confidence = max(0, min(1, confidence))
            
            # Debug: always print the closest match
            closest_name = self.known_names[min_index]
            closest_roll = self.known_roll_numbers[min_index]
            print(f"[DEBUG] Closest match: {closest_name} ({closest_roll}) - Distance: {min_distance:.4f}, Confidence: {confidence:.2%}")
            
            # Recognition decision: ONLY confidence gate (user request)
            # Show green box / accept recognition when confidence >= 85%
            min_confidence_threshold = 0.85

            if confidence >= min_confidence_threshold:
                roll_number = self.known_roll_numbers[min_index]
                name = self.known_names[min_index]
                
                # Check cooldown
                now = datetime.now()
                if roll_number in self.last_detection:
                    time_since_last = (now - self.last_detection[roll_number]).total_seconds()
                    if time_since_last < self.cooldown_seconds:
                        logger.debug(f"Cooldown active for {roll_number}")
                        return None
                
                # Update last detection time
                self.last_detection[roll_number] = now
                
                return {
                    'roll_number': roll_number,
                    'name': name,
                    'distance': float(min_distance),
                    'confidence': float(confidence)
                }
            else:
                print(f"[DEBUG] ❌ Confidence {confidence:.2%} < {min_confidence_threshold*100:.0f}% - NOT RECOGNIZED (distance {min_distance:.4f})")
            
            return None
            
        except Exception as e:
            logger.error(f"Error recognizing face: {str(e)}")
            return None
    
    def detect_and_recognize(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in frame and recognize them
        
        Returns:
            List of recognition results
        """
        try:
            results = []
            
            # Detect faces
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend='opencv',
                enforce_detection=False,
                align=True
            )
            
            for face_data in faces:
                face_area = face_data['facial_area']
                x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']
                
                # Extract face region
                face_region = frame[y:y+h, x:x+w]
                
                # Get embedding
                embedding_objs = DeepFace.represent(
                    img_path=face_region,
                    model_name=self.model_name,
                    detector_backend='skip',  # Already detected
                    enforce_detection=False
                )
                
                if embedding_objs and len(embedding_objs) > 0:
                    embedding = np.array(embedding_objs[0]['embedding'])
                    
                    # Recognize
                    recognition = self.recognize_face(embedding)
                    
                    if recognition:
                        recognition['location'] = (x, y, w, h)
                        results.append(recognition)
            
            return results
            
        except Exception as e:
            logger.debug(f"Detection/recognition error: {str(e)}")
            return []
    
    def save_detection_image(self, frame: np.ndarray, roll_number: str) -> Optional[str]:
        """Save detected face image"""
        try:
            detections_dir = Path("Detections")
            detections_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{roll_number}_{timestamp}.jpg"
            filepath = detections_dir / filename
            
            cv2.imwrite(str(filepath), frame)
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving detection image: {str(e)}")
            return None
    
    def log_detection(self, recognition: Dict, image_path: Optional[str] = None) -> None:
        """Log detection to database"""
        try:
            db.log_detection(
                roll_number=recognition['roll_number'],
                confidence=recognition.get('confidence', 0.0),
                image_path=image_path
            )
            
        except Exception as e:
            logger.error(f"Error logging detection: {str(e)}")
    
    def run_recognition(self, save_detections: bool = True,
                       show_display: bool = True,
                       process_every_n_frames: int = 2) -> None:
        """
        Run real-time face recognition
        
        Args:
            save_detections: Whether to save detection images
            show_display: Whether to show video display
            process_every_n_frames: Process every Nth frame for performance
        """
        try:
            # Load known faces
            if not self.load_known_faces_from_cache():
                logger.info("Cache not found, loading from database...")
                self.load_known_faces_from_database()
            
            if len(self.known_embeddings) == 0:
                logger.error("No registered faces found!")
                logger.info("Please register students first")
                return
            
            # Initialize camera with fallbacks
            cap = self._open_camera()
            if cap is None:
                logger.error("Failed to open camera (tried OpenCV + Picamera2)")
                return
            
            logger.info("Starting face recognition...")
            logger.info("Press 'q' to quit, 'r' to reload cache")
            
            frame_count = 0
            fps_start_time = datetime.now()
            fps = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                frame_count += 1
                display_frame = frame.copy()
                
                # Process every Nth frame
                if frame_count % process_every_n_frames == 0:
                    # Detect all faces first
                    try:
                        all_faces = DeepFace.extract_faces(
                            img_path=frame,
                            detector_backend='opencv',
                            enforce_detection=False,
                            align=False
                        )
                    except Exception as e:
                        all_faces = []
                        logger.debug(f"Face detection error: {str(e)}")

                    # Recognize faces
                    recognitions = self.detect_and_recognize(frame)

                    if len(recognitions) > 0:
                        print(f"\n{'='*60}")

                    # Collect recognized locations for suppression of red boxes
                    recognized_locations = [tuple(r['location']) for r in recognitions]

                    def is_recognized(x, y, w, h, tolerance_xy=8, tolerance_wh=12):
                        for rx, ry, rw, rh in recognized_locations:
                            if (abs(x - rx) <= tolerance_xy and abs(y - ry) <= tolerance_xy and
                                abs(w - rw) <= tolerance_wh and abs(h - rh) <= tolerance_wh):
                                return True
                        return False

                    # Draw boxes: red for unrecognized, green for recognized
                    if show_display:
                        # First draw GREEN boxes for recognized faces
                        for recognition in recognitions:
                            roll_number = recognition['roll_number']
                            name = recognition['name']
                            confidence = recognition.get('confidence', 0)
                            distance = recognition.get('distance', 0)
                            x, y, w, h = recognition['location']

                            # Console/log output
                            print(f"✅ RECOGNIZED: {name}")
                            print(f"   Roll Number: {roll_number}")
                            print(f"   Confidence: {confidence:.2%}")
                            print(f"   Distance: {distance:.4f}")
                            print(f"{'='*60}\n")
                            logger.info(f"DETECTED: {name} ({roll_number}) - Confidence: {confidence:.2f}")

                            # Save detection image and log
                            if save_detections:
                                image_path = self.save_detection_image(frame, roll_number)
                                self.log_detection(recognition, image_path)
                            else:
                                self.log_detection(recognition)

                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                            label = f"{roll_number}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            cv2.rectangle(display_frame, (x, y-32), (x + label_size[0] + 8, y), (0, 255, 0), -1)
                            cv2.putText(display_frame, label, (x + 4, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                        # Then draw RED boxes only for faces not recognized
                        for face_data in all_faces:
                            face_area = face_data.get('facial_area', {})
                            x = face_area.get('x', 0)
                            y = face_area.get('y', 0)
                            w = face_area.get('w', 0)
                            h = face_area.get('h', 0)
                            if w <= 0 or h <= 0:
                                continue
                            if is_recognized(x, y, w, h):
                                # Already drawn green
                                continue
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            logger.debug(f"[DEBUG] Unrecognized face at ({x}, {y}, {w}, {h})")
                    else:
                        # Even if not displaying, still log recognized faces
                        for recognition in recognitions:
                            roll_number = recognition['roll_number']
                            name = recognition['name']
                            confidence = recognition.get('confidence', 0)
                            distance = recognition.get('distance', 0)
                            print(f"✅ RECOGNIZED: {name} - {roll_number} | Conf {confidence:.2%} Dist {distance:.4f}")
                            if save_detections:
                                image_path = self.save_detection_image(frame, roll_number)
                                self.log_detection(recognition, image_path)
                            else:
                                self.log_detection(recognition)
                
                # Calculate FPS
                if frame_count % 30 == 0:
                    fps_end_time = datetime.now()
                    time_diff = (fps_end_time - fps_start_time).total_seconds()
                    fps = 30 / time_diff if time_diff > 0 else 0
                    fps_start_time = fps_end_time
                
                # Display
                if show_display:
                    # Draw info
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Students: {len(self.known_embeddings)}", (20, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Face Recognition', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("Quitting...")
                    break
                elif key == ord('r'):
                    logger.info("Reloading cache...")
                    self.load_known_faces_from_database()
            
            try:
                cap.release()
            except Exception:
                pass
            cv2.destroyAllWindows()
            
        except Exception as e:
            logger.error(f"Error during recognition: {str(e)}")
            raise
        finally:
            if 'cap' in locals():
                try:
                    cap.release()
                except Exception:
                    pass
            cv2.destroyAllWindows()

def main():
    """Main function"""
    
    print("\n" + "="*60)
    print("High-Resolution Face Recognition System (DeepFace)")
    print("="*60)
    
    # Initialize recognition system
    recognizer = FaceRecognitionDeepFace(
        camera_index=0,
        high_res_width=1280,
        high_res_height=720,
        model_name='Facenet512',
        distance_threshold=0.4  # Stricter threshold for 95%+ accuracy (Facenet512 uses cosine similarity)
    )
    
    # Determine if display should be shown based on environment
    show_display = not recognizer.headless
    if recognizer.headless:
        logger.info("Running in headless mode - display disabled")
    else:
        logger.info("Display enabled - showing camera feed")
    
    # Run recognition
    recognizer.run_recognition(
        save_detections=True,
        show_display=show_display,
        process_every_n_frames=2  # Process every 2nd frame for better FPS
    )

if __name__ == "__main__":
    main()

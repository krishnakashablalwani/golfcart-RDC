import os
import sys
import cv2
import numpy as np
import time
from datetime import datetime
from typing import List, Optional, Tuple
from pathlib import Path
import logging
import platform
import argparse
import os.path as osp

# GPU/CPU toggle
USE_GPU = os.environ.get("USE_GPU", "1")  # Default to GPU (CUDA) if available

# Global variables for lazy loading
_insightface = None
_retinaface = None
_providers = ["CPUExecutionProvider"]

def _load_insightface():
    """Lazy load InsightFace model when first needed"""
    global _insightface, _providers
    
    if _insightface is not None:
        return _insightface
    
    try:
        from insightface.model_zoo.arcface_onnx import ArcFaceONNX
        
        if USE_GPU in ("1", "true", "True"):
            _providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            _providers = ["CPUExecutionProvider"]
        
        # Use standalone ArcFace model from buffalo_sc (MobileFaceNet - lighter)
        model_path = osp.join(
            osp.expanduser("~"), ".insightface", "models", "buffalo_sc", "w600k_mbf.onnx"
        )
        
        if osp.exists(model_path):
            print(f"Loading InsightFace model from {model_path}...")
            _insightface = ArcFaceONNX(model_file=model_path)
            _insightface.prepare(
                ctx_id=0 if _providers[0] == "CUDAExecutionProvider" else -1
            )
            print(f"✓ InsightFace loaded: w600k_mbf (MobileFaceNet) with {_providers[0]}")
            return _insightface
        else:
            print(f"⚠ Model file not found at {model_path}")
            print("Please ensure buffalo_sc model is downloaded")
            return None
    except Exception as e:
        print(f"⚠ InsightFace initialization error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Optional: RetinaFace for alignment
try:
    from retina_face import RetinaFace
    _retinaface = RetinaFace
except Exception:
    _retinaface = None

from modules.database import db
from modules.excel_parser import StudentExcelParser

num_samples = 50  # Production setting for 5000+ people (increased from 30)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FaceRegistrationDeepFace:
    def __init__(
        self,
        camera_index: int = 1,
        high_res_width: int = 1920,
        high_res_height: int = 1080,
    ):
        """
        Initialize face registration system with InsightFace

        Args:
            camera_index: Camera device index
            high_res_width: Camera capture width (high resolution)
            high_res_height: Camera capture height (high resolution)
        """
        self.camera_index = camera_index
        self.high_res_width = high_res_width
        self.high_res_height = high_res_height
        self.samples_base_dir = "Samples"

        # Detect headless mode - only on Linux without X11/Wayland display
        display_env = os.environ.get("DISPLAY", "")
        is_linux = platform.system().lower() == "linux"
        self.headless = is_linux and display_env in (None, "", "unknown")

        os.makedirs(self.samples_base_dir, exist_ok=True)

        # Load model when class is instantiated
        _load_insightface()

        logger.info("Face Registration initialized with InsightFace ArcFace")
        logger.info(f"Resolution: {high_res_width}x{high_res_height}")
        logger.info(
            f"Headless mode: {self.headless} (DISPLAY={os.environ.get('DISPLAY', 'NOT SET')})"
        )
        logger.info(f"InsightFace model: w600k_mbf, Providers: {_providers}")

    def detect_face_with_quality_check(
        self, frame: np.ndarray
    ) -> Tuple[bool, Optional[dict], Optional[np.ndarray]]:
        """
        Detect face in frame and check quality

        Returns:
            (is_good_quality, face_info, face_region)
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(150, 150)  # Increased from 100 for better quality
            )

            if len(faces) == 0:
                return False, None, None

            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face

            padding_x = int(w * 0.3)
            padding_y = int(h * 0.3)
            x1 = max(0, x - padding_x)
            y1 = max(0, y - padding_y)
            x2 = min(frame.shape[1], x + w + padding_x)
            y2 = min(frame.shape[0], y + h + padding_y)

            face_region = frame[y1:y2, x1:x2]

            face_info = {"x": x, "y": y, "w": w, "h": h, "confidence": 1.0}

            # Quality checks - RELAXED for better camera compatibility
            is_good = True
            if w < 80 or h < 80:  # Minimum face size
                is_good = False

            brightness = cv2.mean(gray[y : y + h, x : x + w])[0]
            if brightness < 20 or brightness > 240:  # Wide brightness range
                is_good = False

            laplacian_var = cv2.Laplacian(gray[y : y + h, x : x + w], cv2.CV_64F).var()
            if laplacian_var < 20:  # Very relaxed blur check
                is_good = False

            return is_good, face_info, face_region

        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return False, None, None

    def get_face_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding using InsightFace

        Returns:
            512-dimensional embedding or None if failed
        """
        try:
            # Ensure model is loaded
            global _insightface
            if _insightface is None:
                _insightface = _load_insightface()
            
            if _insightface is None:
                logger.error("InsightFace not initialized. Cannot extract embeddings.")
                return None

            # Align to 112x112 if possible
            try:
                if _retinaface:
                    detections = _retinaface.detect_faces(frame)
                    if detections:
                        k = next(iter(detections.keys()))
                        attrs = detections[k]
                        lm = attrs.get("landmarks", {})
                        if lm:
                            src = np.float32(
                                [
                                    lm["left_eye"],
                                    lm["right_eye"],
                                    lm["nose"],
                                    lm["mouth_left"],
                                    lm["mouth_right"],
                                ]
                            )
                            dst = np.float32(
                                [
                                    [38.2946, 51.6963],
                                    [73.5318, 51.5014],
                                    [56.0252, 71.7366],
                                    [41.5493, 92.3655],
                                    [70.7299, 92.2041],
                                ]
                            )
                            M = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)[0]
                            aligned = cv2.warpAffine(frame, M, (112, 112))
                            embedding = _insightface.get_feat(aligned)
                            return np.array(embedding).flatten().astype(np.float32)
            except Exception:
                pass

            # Fallback: simple resize
            face_112 = cv2.resize(frame, (112, 112))
            embedding = _insightface.get_feat(face_112)
            embedding = np.array(embedding).flatten().astype(np.float32)
            
            # L2 normalize to unit norm (CRITICAL!)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding

        except MemoryError as e:
            logger.error(f"Out of memory while extracting embedding: {str(e)}")
            logger.error("Try closing other applications or reducing image size")
            return None
        except Exception as e:
            logger.error(f"Error extracting embedding: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _open_camera(self):
        """Open camera with platform-specific fallbacks"""
        system = platform.system().lower()
        backends = []
        
        if system.startswith("win"):
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

        # Try different camera indices (0, 1, 2)
        for camera_idx in [self.camera_index, 0, 1, 2]:
            for be in backends:
                try:
                    logger.info(f"Trying camera index {camera_idx} with backend {be}")
                    cap = cv2.VideoCapture(camera_idx, be)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.high_res_width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.high_res_height)
                        cap.set(cv2.CAP_PROP_FPS, 30)

                        # Test with a few frames
                        for _ in range(5):
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                logger.info(f"✓ Camera opened: index {camera_idx}, frame size: {frame.shape}")
                                return cap
                        cap.release()
                except Exception as e:
                    logger.debug(f"Camera attempt failed: {e}")
                    continue

        logger.warning("OpenCV camera failed, trying Picamera2 (Raspberry Pi)")
        if system == "linux":
            try:
                from picamera2 import Picamera2

                class Picam2Capture:
                    def __init__(self, width, height):
                        self.picam = Picamera2()

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
                        arr = self.picam.capture_array("main")
                        frame = cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_I420)
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

    def register_student(
        self, roll_number: str, student_name: str, num_samples: int = 50
    ) -> bool:
        """
        Register a student by capturing face samples and creating embedding

        Args:
            roll_number: Student's roll number
            student_name: Student's name
            num_samples: Number of samples to capture

        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Create sample directory
            year = datetime.now().year
            dept_code = roll_number.split("-")[2] if "-" in roll_number else "000"
            sample_dir = Path(self.samples_base_dir) / str(year) / dept_code / roll_number
            sample_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Saving samples to: {sample_dir}")

            cap = self._open_camera()
            if cap is None:
                logger.error("Failed to open camera")
                return False

            time.sleep(2)

            logger.info(f"Starting registration for: {student_name} ({roll_number})")
            logger.info(f"Will capture {num_samples} samples")

            if self.headless:
                logger.info("\n⭐ HEADLESS MODE - Auto-capturing good quality frames")
                logger.info("   - System will automatically capture when face quality is good")
                logger.info("   - Press Ctrl+C to skip this student\n")
            else:
                logger.info("\n⭐ INSTRUCTIONS:")
                logger.info("   - Position student's face in the frame")
                logger.info("   - Wait for GREEN box around face")
                logger.info(f"   - Press 'C' each time to capture ({num_samples} times total)")
                logger.info("   - Press 'Q' to skip this student\n")

            samples_captured = 0
            embeddings = []
            frames_since_last_capture = 0

            while samples_captured < num_samples:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break

                is_good, face_info, face_region = self.detect_face_with_quality_check(frame)

                display_frame = frame.copy()

                if face_info:
                    x, y, w, h = (
                        face_info["x"],
                        face_info["y"],
                        face_info["w"],
                        face_info["h"],
                    )
                    color = (0, 255, 0) if is_good else (0, 0, 255)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 3)

                    status = "GOOD QUALITY" if is_good else "POOR QUALITY"
                    cv2.putText(
                        display_frame,
                        status,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

                cv2.putText(
                    display_frame,
                    f"Samples: {samples_captured}/{num_samples}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    2,
                )

                if not self.headless:
                    if not self.headless:
                        cv2.imshow("Face Registration", display_frame)
                        key = cv2.waitKey(30) & 0xFF
                    else:
                        key = 0xFF
                else:
                    key = 0xFF
                    frames_since_last_capture += 1

                should_capture = False
                if self.headless:
                    if (
                        is_good
                        and face_region is not None
                        and frames_since_last_capture >= 30
                    ):
                        should_capture = True
                        frames_since_last_capture = 0
                else:
                    if key == ord("c") or key == ord("C"):
                        if is_good and face_region is not None:
                            should_capture = True
                        else:
                            logger.warning(
                                "Cannot capture - face quality is poor or not detected"
                            )

                if should_capture:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    sample_filename = sample_dir / f"{roll_number}_{samples_captured+1}_{timestamp}.jpg"
                    cv2.imwrite(str(sample_filename), face_region)

                    embedding = self.get_face_embedding(face_region)
                    if embedding is not None:
                        embeddings.append(embedding)
                        samples_captured += 1
                        logger.info(
                            f"✓ Sample {samples_captured}/{num_samples} captured and processed"
                        )

                        if not self.headless:
                            cv2.putText(
                                display_frame,
                                f"CAPTURED {samples_captured}/{num_samples}",
                                (
                                    int(frame.shape[1] / 2) - 200,
                                    int(frame.shape[0] / 2),
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,
                                (0, 255, 0),
                                3,
                            )
                            if not self.headless:
                                cv2.imshow("Face Registration", display_frame)
                                cv2.waitKey(500)
                    else:
                        logger.warning("Failed to extract embedding - try again")

                if not self.headless and (key == ord("q") or key == ord("Q")):
                    logger.info("Registration cancelled by user")
                    break

            try:
                cap.release()
            except Exception:
                pass
            if not self.headless:
                cv2.destroyAllWindows()

            if samples_captured == num_samples and len(embeddings) == num_samples:
                embeddings_array = np.array(embeddings)
                
                # Use L2 distance for outlier detection (same as recognition)
                from scipy.spatial.distance import pdist, squareform
                try:
                    distances = squareform(pdist(embeddings_array, metric='euclidean'))
                    # For each embedding, get mean distance to others
                    mean_distances = np.mean(distances, axis=1)
                    # Remove embeddings with distance > 1.5 * median (outliers)
                    median_dist = np.median(mean_distances)
                    valid_indices = np.where(mean_distances <= median_dist * 1.5)[0]
                    
                    if len(valid_indices) >= max(30, num_samples * 0.6):
                        # Keep majority of samples
                        filtered_embeddings = embeddings_array[valid_indices]
                        logger.info(f"Filtered embeddings: {len(valid_indices)}/{num_samples} kept (L2 outliers removed)")
                        logger.info(f"Median L2 distance between samples: {median_dist:.4f}")
                        avg_embedding = np.mean(filtered_embeddings, axis=0)
                    else:
                        # Keep all if filtering removes too many
                        logger.warning("Too many outliers detected, keeping all samples")
                        avg_embedding = np.mean(embeddings_array, axis=0)
                except Exception as e:
                    logger.warning(f"Outlier filtering failed: {e}, using all samples")
                    avg_embedding = np.mean(embeddings_array, axis=0)
                
                # L2 normalize the averaged embedding to unit norm (CRITICAL!)
                norm = np.linalg.norm(avg_embedding)
                if norm > 0:
                    avg_embedding = avg_embedding / norm
                
                avg_embedding = avg_embedding.astype(np.float32)
                
                logger.info(f"Master embedding norm: {np.linalg.norm(avg_embedding):.4f}")

                success = db.store_face_embedding(
                    roll_number=roll_number,
                    embedding=avg_embedding.tolist(),
                    sample_paths=[
                        str(sample_dir / f"{roll_number}_{i+1}_*.jpg")
                        for i in range(num_samples)
                    ],
                    num_samples=num_samples,
                )

                if success:
                    logger.info(f"Successfully registered {student_name} ({roll_number})")
                    logger.info(f"Samples saved to: {sample_dir}")

                    try:
                        if db.mark_student_registered(roll_number):
                            logger.info("Marked student as registered in database")
                        else:
                            logger.warning("Failed to mark student as registered")
                    except Exception as e:
                        logger.warning(f"Could not mark as registered: {str(e)}")

                    return True
                else:
                    logger.error("Failed to store face embedding in database")
                    return False
            else:
                logger.warning(
                    f"Incomplete registration: {samples_captured}/{num_samples} samples"
                )
                return False

        except Exception as e:
            logger.error(f"Error during registration: {str(e)}")
            return False
        finally:
            if "cap" in locals():
                try:
                    cap.release()
                except Exception:
                    pass
            if not self.headless:
                cv2.destroyAllWindows()

    def batch_register(self, roll_numbers: List[str]) -> dict:
        """
        Register multiple students

        Args:
            roll_numbers: List of roll numbers to register

        Returns:
            Dictionary with success/failure counts
        """
        results = {"success": 0, "failed": 0, "failed_students": []}

        for i, roll_number in enumerate(roll_numbers):
            logger.info(f"\n=== Registering student {i+1}/{len(roll_numbers)} ===")

            student = db.get_student(roll_number)
            if not student:
                logger.error(f"Student {roll_number} not found")
                results["failed"] += 1
                results["failed_students"].append(roll_number)
                continue

            student_name = student.get("name", "Unknown")

            print(f"\n{'='*60}")
            print(f"Student {i+1}/{len(roll_numbers)}")
            print(f"Name: {student_name}")
            print(f"Roll Number: {roll_number}")
            print(f"{'='*60}\n")

            response = input("Register this student? (y/n/skip remaining): ").strip().lower()

            if response == "skip remaining" or response == "s":
                logger.info("Skipping remaining students")
                break
            elif response != "y":
                logger.info(f"Skipped {student_name}")
                continue

            success = self.register_student(roll_number, student_name)

            if success:
                results["success"] += 1
                print(f"✅ Successfully registered {student_name}\n")
            else:
                results["failed"] += 1
                results["failed_students"].append(roll_number)
                print(f"❌ Failed to register {student_name}\n")

        return results


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="High-Resolution Face Registration System")
    parser.add_argument("--no-display", action="store_true", help="Disable camera display (headless mode)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("High-Resolution Face Registration System (InsightFace)")
    print("=" * 60)
    
    # Initialize registrar
    registrar = FaceRegistrationDeepFace(
        camera_index=0,
        high_res_width=1920,
        high_res_height=1080,
    )
    
    # Override headless detection if --no-display flag is set
    if args.no_display:
        registrar.headless = True
        logger.info("Display disabled via --no-display flag")

    print("\nLoading unregistered students from database...")
    unregistered_students = db.list_unregistered_students()

    if not unregistered_students:
        print("\n✅ All students are already registered!")
        print("No action needed.\n")
        return

    specific_roll = sys.argv[1] if len(sys.argv) > 1 else None

    if specific_roll:
        student = db.get_student(specific_roll)
        if student:
            students_to_register = [specific_roll]
            print(f"\n📋 Registering specific student: {specific_roll}")
        else:
            print(f"\n❌ Student {specific_roll} not found in database")
            return
    else:
        students_to_register = [s["roll_number"] for s in unregistered_students]

        print(f"\n📋 Found {len(students_to_register)} students to register")
        print("-" * 60)
        for i, s in enumerate(unregistered_students, 1):
            print(f"{i}. {s['name']} - {s['roll_number']}")
        print("-" * 60)

    input("\nPress ENTER to start registration...")

    results = registrar.batch_register(students_to_register)

    print("\n" + "=" * 60)
    print("REGISTRATION SUMMARY")
    print("=" * 60)
    print(f"Total: {len(students_to_register)}")
    print(f"Success: {results['success']}")
    print(f"Failed: {results['failed']}")
    print(f"Remaining: {len(students_to_register) - results['success'] - results['failed']}")
    print("=" * 60)

    if results["failed_students"]:
        print("\nFailed Students:")
        for roll in results["failed_students"]:
            print(f"  - {roll}")

    print()


if __name__ == "__main__":
    main()

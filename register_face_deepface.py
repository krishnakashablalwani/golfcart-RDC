"""
High-Resolution Face Registration System using DeepFace
Captures 15 face samples per student and stores them with embeddings in MongoDB
Optimized for larger distance detection
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
import os
import time
from datetime import datetime
from typing import List, Optional, Tuple
from pathlib import Path
from deepface import DeepFace
import logging
import platform

from modules.database import db
from modules.excel_parser import StudentExcelParser


num_samples = 15  # Number of samples to capture per student

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FaceRegistrationDeepFace:
    def __init__(self, camera_index: int = 1, 
                 high_res_width: int = 1280, 
                 high_res_height: int = 720,
                 model_name: str = 'Facenet512'):
        """
        Initialize face registration system
        
        Args:
            camera_index: Camera device index
            high_res_width: Camera capture width (high resolution)
            high_res_height: Camera capture height (high resolution)
            model_name: DeepFace model ('Facenet512' recommended for accuracy)
        """
        self.camera_index = camera_index
        self.high_res_width = high_res_width
        self.high_res_height = high_res_height
        self.model_name = model_name
        self.samples_base_dir = "Samples"
        self.headless = os.environ.get('DISPLAY') in (None, '', 'unknown')
        
        # Create base directory
        os.makedirs(self.samples_base_dir, exist_ok=True)
        
        logger.info(f"Face Registration initialized with {model_name} model")
        logger.info(f"Resolution: {high_res_width}x{high_res_height}")
        logger.info(f"Headless mode: {self.headless} (DISPLAY={os.environ.get('DISPLAY', 'NOT SET')})")
    
    def detect_face_with_quality_check(self, frame: np.ndarray) -> Tuple[bool, Optional[dict], Optional[np.ndarray]]:
        """
        Detect face in frame and check quality
        
        Returns:
            (is_good_quality, face_info, face_region)
        """
        try:
            # Use DeepFace to detect faces
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend='opencv',
                enforce_detection=False,
                align=True
            )
            
            if not faces or len(faces) == 0:
                return False, None, None
            
            # Get the largest face (closest to camera)
            largest_face = max(faces, key=lambda x: x['facial_area']['w'] * x['facial_area']['h'])
            face_area = largest_face['facial_area']
            
            # Extract face region with padding to capture extreme sides
            x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']
            
            # Add 30% padding on all sides to capture full head including sides
            padding_x = int(w * 0.3)
            padding_y = int(h * 0.3)
            
            x_padded = max(0, x - padding_x)
            y_padded = max(0, y - padding_y)
            w_padded = min(frame.shape[1] - x_padded, w + 2 * padding_x)
            h_padded = min(frame.shape[0] - y_padded, h + 2 * padding_y)
            
            face_region = frame[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
            
            # Quality checks (stricter for high accuracy)
            min_face_width = 100  # Minimum face width for clear features
            min_face_height = 100  # Minimum face height for clear features
            max_face_width = frame.shape[1] * 0.8  # Max 80% of frame
            max_face_height = frame.shape[0] * 0.8
            
            # Check face size
            if w < min_face_width or h < min_face_height:
                return False, {"reason": "Face too small - move closer"}, None
            
            if w > max_face_width or h > max_face_height:
                return False, {"reason": "Face too large - move back"}, None
            
            # Check face position (should be well-centered)
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            center_threshold = 200  # pixels from center (stricter centering)
            if abs(face_center_x - frame_center_x) > center_threshold or \
               abs(face_center_y - frame_center_y) > center_threshold:
                return False, {"reason": "Center your face in frame"}, None
            
            # Check image sharpness (Laplacian variance) - stricter
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if laplacian_var < 50:  # Require sharp images
                return False, {"reason": "Image blurry - hold still"}, None
            
            # Check brightness - stricter range
            brightness = np.mean(gray)
            if brightness < 50:  # Good lighting required
                return False, {"reason": "Too dark - need better lighting"}, None
            if brightness > 200:  # Avoid overexposure
                return False, {"reason": "Too bright - reduce lighting"}, None
            
            # Return success with face coordinates
            return True, {"x": x, "y": y, "w": w, "h": h}, face_region
            
        except Exception as e:
            logger.debug(f"Face detection failed: {str(e)}")
            return False, None, None
    
    def get_face_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding using DeepFace
        
        Returns:
            128/512-dimensional embedding or None if failed
        """
        try:
            # Get embedding
            embedding_objs = DeepFace.represent(
                img_path=frame,
                model_name=self.model_name,
                detector_backend='opencv',
                enforce_detection=False,
                align=True
            )
            
            if not embedding_objs or len(embedding_objs) == 0:
                return None
            
            # Get the first face's embedding
            embedding = np.array(embedding_objs[0]['embedding'])
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {str(e)}")
            return None
    
    def create_sample_directory(self, roll_number: str) -> Path:
        """
        Create directory structure: Samples/Year/Department/RollNumber
        
        Args:
            roll_number: Format COLLEGE-YY-DEPT-ROLL (e.g., 2451-25-733-075)
        
        Returns:
            Path to student's sample directory
        """
        try:
            parts = roll_number.split('-')
            if len(parts) != 4:
                raise ValueError(f"Invalid roll number format: {roll_number}")
            
            college, year, dept, roll = parts
            
            # Create directory structure
            student_dir = Path(self.samples_base_dir) / f"20{year}" / dept / roll_number
            student_dir.mkdir(parents=True, exist_ok=True)
            
            return student_dir
            
        except Exception as e:
            logger.error(f"Error creating directory: {str(e)}")
            raise
    
    def _open_camera(self):
        """Open camera with platform-specific fallbacks, including Picamera2 on Raspberry Pi."""
        system = platform.system().lower()
        # Try OpenCV backends
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

        # Fallback to Picamera2 on Linux (Raspberry Pi)
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
                        # Capture YUV420 and convert to BGR
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

    def register_student(self, roll_number: str, student_name: str, 
                        num_samples: int = num_samples) -> bool:
        """
        Register a student by capturing face samples
        
        Args:
            roll_number: Student's roll number
            student_name: Student's name
            num_samples: Number of samples to capture (default: 15)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if student exists in database
            student = db.get_student(roll_number)
            if not student:
                logger.error(f"Student {roll_number} not found in database")
                logger.info("Please import students from Excel first")
                return False
            
            # Create sample directory
            sample_dir = self.create_sample_directory(roll_number)
            logger.info(f"Saving samples to: {sample_dir}")
            
            # Initialize camera with fallbacks
            cap = self._open_camera()
            if cap is None:
                logger.error("Failed to open camera (tried OpenCV + Picamera2)")
                return False
            
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
                logger.info("   - Press 'C' each time to capture (15 times total)")
                logger.info("   - Press 'Q' to skip this student\n")
            
            samples_captured = 0
            frames_since_last_capture = 0
            embeddings = []
            
            while samples_captured < num_samples:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Detect face and check quality
                is_good, face_info, face_region = self.detect_face_with_quality_check(frame)
                
                # Draw UI only if not headless
                if not self.headless:
                    display_frame = frame.copy()
                
                if is_good and face_info:
                    if not self.headless:
                        # Draw green rectangle around face
                        x, y, w, h = face_info['x'], face_info['y'], face_info['w'], face_info['h']
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                        
                        # Show status
                        cv2.putText(display_frame, "READY - Press 'C' to capture", 
                                  (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    if not self.headless:
                        # Show reason for bad quality
                        reason = face_info.get('reason', 'No face detected') if face_info else 'No face detected'
                        cv2.putText(display_frame, reason, 
                                  (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.putText(display_frame, "Press 'C' to capture anyway", 
                                  (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if not self.headless:
                    # Show progress
                    progress_text = f"Samples: {samples_captured}/{num_samples}"
                    cv2.putText(display_frame, progress_text, 
                              (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Show student info
                    cv2.putText(display_frame, f"Student: {student_name}", 
                              (20, display_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Roll No: {roll_number}", 
                              (20, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (255, 255, 255), 2)
                    
                    cv2.imshow('Face Registration', display_frame)
                    key = cv2.waitKey(30) & 0xFF
                else:
                    key = 0xFF  # No key in headless mode
                    frames_since_last_capture += 1
                
                # Capture logic: manual in GUI mode, auto in headless mode
                should_capture = False
                if self.headless:
                    # Auto-capture in headless: good quality + 30 frames gap (1 sec)
                    if is_good and face_region is not None and frames_since_last_capture >= 30:
                        should_capture = True
                        frames_since_last_capture = 0
                else:
                    # Manual capture in GUI mode
                    if (key == ord('c') or key == ord('C')) and face_region is not None:
                        should_capture = True
                
                if should_capture:
                    # Get embedding
                    embedding = self.get_face_embedding(frame)
                    
                    if embedding is not None:
                        # Save image
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"{roll_number}_{samples_captured+1}_{timestamp}.jpg"
                        filepath = sample_dir / filename
                        cv2.imwrite(str(filepath), frame)
                        
                        # Store embedding
                        embeddings.append(embedding)
                        
                        samples_captured += 1
                        logger.info(f"✅ Captured sample {samples_captured}/{num_samples}")
                        
                        if not self.headless:
                            # Brief visual feedback
                            cv2.putText(display_frame, f"CAPTURED {samples_captured}!", 
                                      (display_frame.shape[1]//2 - 150, display_frame.shape[0]//2), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                            cv2.imshow('Face Registration', display_frame)
                            cv2.waitKey(500)  # Show feedback for 500ms
                    else:
                        logger.warning("Failed to extract embedding - try again")
                
                # Quit on Q key (GUI mode only)
                if not self.headless and (key == ord('q') or key == ord('Q')):
                    logger.info("Registration cancelled by user")
                    break
            
            try:
                cap.release()
            except Exception:
                pass
            cv2.destroyAllWindows()
            
            if samples_captured == num_samples and len(embeddings) == num_samples:
                # Calculate average embedding
                avg_embedding = np.mean(embeddings, axis=0)
                
                # Store in database
                success = db.store_face_embedding(
                    roll_number=roll_number,
                    embedding=avg_embedding.tolist(),
                    sample_paths=[str(sample_dir / f"{roll_number}_{i+1}_*.jpg") 
                                 for i in range(num_samples)],
                    num_samples=num_samples
                )
                
                if success:
                    logger.info(f"Successfully registered {student_name} ({roll_number})")
                    logger.info(f"Samples saved to: {sample_dir}")
                    # Mark student as registered in DB and Excel
                    try:
                        if db.mark_student_registered(roll_number):
                            logger.info("Marked student as registered in database")
                        else:
                            logger.warning("Failed to mark student as registered in database")
                    except Exception as e:
                        logger.warning(f"Database register mark error: {e}")

                    try:
                        parser = StudentExcelParser()
                        if parser.mark_registered_in_excel(roll_number):
                            logger.info("Updated Excel: registered=YES for student")
                        else:
                            logger.warning("Failed to update Excel registered flag for student")
                    except Exception as e:
                        logger.warning(f"Excel update error: {e}")
                    return True
                else:
                    logger.error("Failed to store in database")
                    return False
            else:
                logger.warning(f"Incomplete registration: {samples_captured}/{num_samples} samples")
                return False
                
        except Exception as e:
            logger.error(f"Error during registration: {str(e)}")
            return False
        finally:
            if 'cap' in locals():
                try:
                    cap.release()
                except Exception:
                    pass
            cv2.destroyAllWindows()
    
    def batch_register(self, roll_numbers: List[str]) -> dict:
        """
        Register multiple students
        
        Args:
            roll_numbers: List of roll numbers to register
        
        Returns:
            Dictionary with success/failure counts
        """
        results = {
            'success': 0,
            'failed': 0,
            'failed_students': []
        }
        
        for i, roll_number in enumerate(roll_numbers):
            logger.info(f"\n=== Registering student {i+1}/{len(roll_numbers)} ===")
            
            student = db.get_student(roll_number)
            if not student:
                logger.error(f"Student {roll_number} not found")
                results['failed'] += 1
                results['failed_students'].append(roll_number)
                continue
            
            student_name = student.get('name', 'Unknown')
            
            success = self.register_student(roll_number, student_name)
            
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1
                results['failed_students'].append(roll_number)
        
        return results

def main():
    """Main function - load all unregistered students and register them"""
    import sys
    
    print("\n" + "="*60)
    print("High-Resolution Face Registration System (DeepFace)")
    print("="*60)
    
    # Check if specific roll number provided as argument
    specific_roll = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Get all unregistered students
    print("\nLoading unregistered students from database...")
    unregistered = db.list_unregistered_students()
    
    if not unregistered:
        print("\n✅ All students are already registered!")
        return
    
    # Filter to specific student if provided
    if specific_roll:
        unregistered = [s for s in unregistered if s['roll_number'] == specific_roll]
        if not unregistered:
            print(f"\n❌ Student {specific_roll} not found or already registered!")
            return
        print(f"\n🎯 Registering specific student: {specific_roll}")
    
    print(f"\n📋 Found {len(unregistered)} students to register")
    print("-"*60)
    
    # Show list
    for i, student in enumerate(unregistered[:10], 1):
        print(f"{i}. {student['name']} - {student['roll_number']}")
    
    if len(unregistered) > 10:
        print(f"... and {len(unregistered) - 10} more students")
    
    print("-"*60)
    
    # Skip prompt if specific roll number or if headless
    headless = os.environ.get('DISPLAY') in (None, '', 'unknown')
    if not specific_roll and not headless:
        input("\nPress ENTER to start registration...")
    elif not specific_roll:
        print("\n⚠️  BATCH MODE: Will attempt to register all students automatically")
        print("   Place each student in front of camera for 10 samples")
        print("   Press Ctrl+C to skip a student\n")
        import time
        time.sleep(2)
    
    # Initialize registration system
    # Prefer a safer resolution on Raspberry Pi for Picamera2
    if platform.system().lower() == 'linux':
        width, height = 1280, 720
    else:
        width, height = 1920, 1080

    registrar = FaceRegistrationDeepFace(
        camera_index=0,
        high_res_width=width,
        high_res_height=height,
        model_name='Facenet512'
    )
    
    # Register each student
    success_count = 0
    failed_count = 0
    
    for i, student in enumerate(unregistered, 1):
        roll_number = student['roll_number']
        student_name = student.get('name', 'Unknown')
        
        print(f"\n{'='*60}")
        print(f"Student {i}/{len(unregistered)}")
        print(f"Name: {student_name}")
        print(f"Roll Number: {roll_number}")
        print(f"{'='*60}")
        
        # Auto-proceed in headless or specific roll mode
        if specific_roll or headless:
            response = 'y'
            print("\nAuto-registering in batch/headless mode...")
        else:
            # Ask if user wants to register this student
            response = input("\nRegister this student? (y/n/skip remaining): ").strip().lower()
        
        if response == 'skip remaining':
            print("\nSkipping remaining students...")
            break
        elif response == 'n':
            print("Skipped.")
            continue
        
        # Register the student
        success = registrar.register_student(roll_number, student_name)
        
        if success:
            success_count += 1
            print(f"✅ Successfully registered {student_name}")
        else:
            failed_count += 1
            print(f"❌ Failed to register {student_name}")
    
    # Summary
    print(f"\n{'='*60}")
    print("REGISTRATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {len(unregistered)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Remaining: {len(unregistered) - success_count - failed_count}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

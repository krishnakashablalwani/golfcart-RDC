"""
Configuration Module for Golf Cart Face Recognition System
Loads settings from environment variables with defaults
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the face recognition system"""
    
    # MongoDB Configuration
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    DB_NAME = os.getenv('DB_NAME', 'golfcart_face_recognition')
    
    # Camera Configuration
    CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
    CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', '1920'))
    CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', '1080'))
    CAMERA_FPS = int(os.getenv('CAMERA_FPS', '30'))
    
    # Face Recognition Settings
    RECOGNITION_THRESHOLD = float(os.getenv('RECOGNITION_THRESHOLD', '0.5'))
    MIN_FACE_SIZE = int(os.getenv('MIN_FACE_SIZE', '80'))
    SAMPLES_PER_STUDENT = int(os.getenv('SAMPLES_PER_STUDENT', '15'))
    
    # Detection Settings
    DETECTION_COOLDOWN_SECONDS = int(os.getenv('DETECTION_COOLDOWN_SECONDS', '30'))
    PROCESS_EVERY_N_FRAMES = int(os.getenv('PROCESS_EVERY_N_FRAMES', '2'))
    
    # File Paths
    EXCEL_FILE = os.getenv('EXCEL_FILE', 'Student information.xlsx')
    DEPARTMENTS_FILE = os.getenv('DEPARTMENTS_FILE', 'departments.txt')
    SAMPLES_DIR = os.getenv('SAMPLES_DIR', 'Samples')
    DETECTIONS_DIR = os.getenv('DETECTIONS_DIR', 'Detections')
    
    # Email Configuration
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
    SENDER_EMAIL = os.getenv('SENDER_EMAIL', '')
    SENDER_PASSWORD = os.getenv('SENDER_PASSWORD', '')
    HOD_EMAIL = os.getenv('HOD_EMAIL', '')
    PRINCIPAL_EMAIL = os.getenv('PRINCIPAL_EMAIL', '')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'golfcart_recognition.log')
    
    @classmethod
    def display_config(cls):
        """Display current configuration"""
        print("\n" + "="*60)
        print("SYSTEM CONFIGURATION")
        print("="*60)
        
        print("\nDatabase:")
        print(f"  URI: {cls.MONGODB_URI}")
        print(f"  Database: {cls.DB_NAME}")
        
        print("\nCamera:")
        print(f"  Index: {cls.CAMERA_INDEX}")
        print(f"  Resolution: {cls.CAMERA_WIDTH}x{cls.CAMERA_HEIGHT}")
        print(f"  FPS: {cls.CAMERA_FPS}")
        
        print("\nRecognition:")
        print(f"  Threshold: {cls.RECOGNITION_THRESHOLD}")
        print(f"  Min Face Size: {cls.MIN_FACE_SIZE}px")
        print(f"  Samples per Student: {cls.SAMPLES_PER_STUDENT}")
        
        print("\nDetection:")
        print(f"  Cooldown: {cls.DETECTION_COOLDOWN_SECONDS}s")
        print(f"  Process Every N Frames: {cls.PROCESS_EVERY_N_FRAMES}")
        
        print("\nFiles:")
        print(f"  Excel: {cls.EXCEL_FILE}")
        print(f"  Departments: {cls.DEPARTMENTS_FILE}")
        print(f"  Samples Dir: {cls.SAMPLES_DIR}")
        print(f"  Detections Dir: {cls.DETECTIONS_DIR}")
        
        print("\nEmail:")
        print(f"  SMTP: {cls.SMTP_SERVER}:{cls.SMTP_PORT}")
        print(f"  Sender: {cls.SENDER_EMAIL if cls.SENDER_EMAIL else 'Not configured'}")
        print(f"  HOD: {cls.HOD_EMAIL if cls.HOD_EMAIL else 'Not configured'}")
        
        print("="*60 + "\n")
    
    @classmethod
    def validate_config(cls):
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check critical settings
        if not os.path.exists(cls.EXCEL_FILE):
            issues.append(f"Excel file not found: {cls.EXCEL_FILE}")
        
        # Check camera settings
        if cls.CAMERA_WIDTH < 640:
            issues.append(f"Camera width too low: {cls.CAMERA_WIDTH} (minimum 640)")
        
        if cls.CAMERA_HEIGHT < 480:
            issues.append(f"Camera height too low: {cls.CAMERA_HEIGHT} (minimum 480)")
        
        # Check recognition threshold
        if not 0.0 <= cls.RECOGNITION_THRESHOLD <= 1.0:
            issues.append(f"Invalid recognition threshold: {cls.RECOGNITION_THRESHOLD} (must be 0.0-1.0)")
        
        # Check samples
        if cls.SAMPLES_PER_STUDENT < 5:
            issues.append(f"Too few samples per student: {cls.SAMPLES_PER_STUDENT} (minimum 5)")
        
        return issues

# Global config instance
config = Config()

if __name__ == "__main__":
    # Display configuration
    config.display_config()
    
    # Validate
    issues = config.validate_config()
    if issues:
        print("Configuration Issues:")
        for issue in issues:
            print(f"  ⚠ {issue}")
    else:
        print("✓ Configuration is valid")

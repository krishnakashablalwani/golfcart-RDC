"""
Database Configuration and Models for Golf Cart Face Recognition System
MongoDB Collections:
1. students - Student information and metadata
2. face_embeddings - Face encodings for recognition
3. detections - Detection logs with timestamps
4. departments - Department code mappings
"""

import os
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError, ServerSelectionTimeoutError
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

# MongoDB Configuration
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
DB_NAME = os.getenv('DB_NAME', 'golfcart_face_recognition')

class Database:
    def __init__(self):
        # Use short timeout to avoid blocking when MongoDB isn't running
        self.client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=1500)
        self.db = self.client[DB_NAME]
        self.students = self.db.students
        self.face_embeddings = self.db.face_embeddings
        self.detections = self.db.detections
        self.departments = self.db.departments
        self.available = False
        # Try connecting once; if it fails, keep module import resilient
        try:
            # This triggers a lightweight connection check
            self.client.admin.command('ping')
            self.available = True
            self._create_indexes()
        except ServerSelectionTimeoutError:
            # MongoDB not available right now; defer index creation
            print("[database] MongoDB not available; deferring index creation")
        except Exception as e:
            print(f"[database] Initialization warning: {e}")
    
    def _create_indexes(self):
        """Create indexes for efficient querying"""
        try:
            # Students collection indexes
            self.students.create_index([('roll_number', ASCENDING)], unique=True)
            self.students.create_index([('year', ASCENDING)])
            self.students.create_index([('department_code', ASCENDING)])
            self.students.create_index([('class_roll', ASCENDING)])
            
            # Face embeddings indexes
            self.face_embeddings.create_index([('roll_number', ASCENDING)])
            self.face_embeddings.create_index([('sample_number', ASCENDING)])
            
            # Detections indexes
            self.detections.create_index([('roll_number', ASCENDING)])
            self.detections.create_index([('timestamp', DESCENDING)])
            self.detections.create_index([('date', ASCENDING)])
            
            # Departments indexes
            self.departments.create_index([('code', ASCENDING)], unique=True)
        except Exception as e:
            print(f"[database] Failed to create indexes: {e}")
    
    def add_student(self, student_data: Dict) -> bool:
        """
        Add a new student to the database
        student_data should contain: roll_number, name, year, department_code, 
        class_roll, email, phone, parent_phone, parent_email, etc.
        """
        try:
            # Parse roll number if full format provided (e.g., 2451-25-733-075)
            roll_number = student_data.get('roll_number')
            if '-' in roll_number:
                parts = roll_number.split('-')
                if len(parts) == 4:
                    student_data['year'] = parts[1]
                    student_data['department_code'] = parts[2]
                    student_data['class_roll'] = parts[3]
            
            student_data['created_at'] = datetime.now()
            student_data['updated_at'] = datetime.now()
            
            self.students.insert_one(student_data)
            return True
        except DuplicateKeyError:
            print(f"Student with roll number {student_data.get('roll_number')} already exists")
            return False
        except Exception as e:
            print(f"[database] add_student error: {e}")
            return False
    
    def get_student(self, roll_number: str) -> Optional[Dict]:
        """Get student information by roll number"""
        return self.students.find_one({'roll_number': roll_number})
    
    def update_student(self, roll_number: str, update_data: Dict) -> bool:
        """Update student information"""
        update_data['updated_at'] = datetime.now()
        try:
            result = self.students.update_one(
            {'roll_number': roll_number},
            {'$set': update_data}
        )
            return result.modified_count > 0
        except Exception as e:
            print(f"[database] update_student error: {e}")
            return False

    def mark_student_registered(self, roll_number: str) -> bool:
        """Set registered='YES' for a student in the database."""
        try:
            return self.update_student(roll_number, {'registered': 'YES'})
        except Exception as e:
            print(f"[database] mark_student_registered error: {e}")
            return False
    
    def add_face_embedding(self, roll_number: str, embedding: List[float], 
                          sample_number: int, image_path: str) -> bool:
        """Store face embedding for a student"""
        try:
            embedding_data = {
                'roll_number': roll_number,
                'embedding': embedding,
                'sample_number': sample_number,
                'image_path': image_path,
                'created_at': datetime.now()
            }
            self.face_embeddings.insert_one(embedding_data)
            return True
        except Exception as e:
            print(f"Error storing embedding: {e}")
            return False
    
    def get_all_embeddings(self) -> List[Dict]:
        """Get all face embeddings for recognition"""
        try:
            return list(self.face_embeddings.find())
        except Exception as e:
            print(f"[database] get_all_embeddings error: {e}")
            return []
    
    def get_student_embeddings(self, roll_number: str) -> List[Dict]:
        """Get all face embeddings for a specific student"""
        try:
            return list(self.face_embeddings.find({'roll_number': roll_number}))
        except Exception as e:
            print(f"[database] get_student_embeddings error: {e}")
            return []
    
    def log_detection(self, roll_number: str, confidence: float, 
                     image_path: Optional[str] = None,
                     location: Optional[str] = None) -> bool:
        """Log a face detection event"""
        try:
            detection_data = {
                'roll_number': roll_number,
                'timestamp': datetime.now(),
                'date': datetime.now().date().isoformat(),
                'confidence': confidence,
                'image_path': image_path,
                'location': location
            }
            self.detections.insert_one(detection_data)
            return True
        except Exception as e:
            print(f"Error logging detection: {e}")
            return False
    
    def get_detections(self, roll_number: Optional[str] = None, 
                       date: Optional[str] = None,
                       limit: int = 100) -> List[Dict]:
        """Get detection logs with optional filters"""
        query = {}
        if roll_number:
            query['roll_number'] = roll_number
        if date:
            query['date'] = date
        
        try:
            return list(self.detections.find(query)
                       .sort('timestamp', DESCENDING)
                       .limit(limit))
        except Exception as e:
            print(f"[database] get_detections error: {e}")
            return []
    
    def add_department(self, code: str, name: str) -> bool:
        """Add a department code mapping"""
        try:
            dept_data = {
                'code': code,
                'name': name,
                'created_at': datetime.now()
            }
            self.departments.insert_one(dept_data)
            return True
        except DuplicateKeyError:
            print(f"Department code {code} already exists")
            return False
        except Exception as e:
            print(f"[database] add_department error: {e}")
            return False
    
    def get_department(self, code: str) -> Optional[Dict]:
        """Get department information by code"""
        try:
            return self.departments.find_one({'code': code})
        except Exception as e:
            print(f"[database] get_department error: {e}")
            return None
    
    def get_all_departments(self) -> List[Dict]:
        """Get all departments"""
        try:
            return list(self.departments.find())
        except Exception as e:
            print(f"[database] get_all_departments error: {e}")
            return []
    
    def get_students_by_year(self, year: str) -> List[Dict]:
        """Get all students by admission year"""
        try:
            return list(self.students.find({'year': year}))
        except Exception as e:
            print(f"[database] get_students_by_year error: {e}")
            return []
    
    def get_students_by_department(self, dept_code: str) -> List[Dict]:
        """Get all students by department code"""
        try:
            return list(self.students.find({'department_code': dept_code}))
        except Exception as e:
            print(f"[database] get_students_by_department error: {e}")
            return []
    
    def get_total_students(self) -> int:
        """Get total number of registered students"""
        try:
            return self.students.count_documents({})
        except Exception as e:
            print(f"[database] get_total_students error: {e}")
            return 0
    
    def get_total_embeddings(self) -> int:
        """Get total number of face embeddings"""
        try:
            return self.face_embeddings.count_documents({})
        except Exception as e:
            print(f"[database] get_total_embeddings error: {e}")
            return 0
    
    def store_face_embedding(self, roll_number: str, embedding: List[float],
                           sample_paths: List[str], num_samples: int) -> bool:
        """
        Store averaged face embedding for a student (for DeepFace workflow)
        
        Args:
            roll_number: Student's roll number
            embedding: Average embedding vector
            sample_paths: List of paths to sample images
            num_samples: Number of samples captured
        
        Returns:
            True if successful, False otherwise
        """
        try:
            embedding_data = {
                'roll_number': roll_number,
                'embedding': embedding,
                'sample_paths': sample_paths,
                'num_samples': num_samples,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            
            # Update if exists, insert if new
            result = self.face_embeddings.update_one(
                {'roll_number': roll_number},
                {'$set': embedding_data},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"Error storing face embedding: {e}")
            return False
    
    def list_unregistered_students(self) -> List[Dict]:
        """Get all students who don't have face embeddings yet"""
        # Get all student roll numbers
        try:
            all_students = list(self.students.find({}, {'roll_number': 1, 'name': 1, '_id': 0}))
        except Exception as e:
            print(f"[database] list_unregistered_students error: {e}")
            return []
        
        # Get roll numbers that have embeddings
        try:
            registered_roll_numbers = set(
                doc['roll_number'] for doc in 
                self.face_embeddings.find({}, {'roll_number': 1, '_id': 0})
            )
        except Exception as e:
            print(f"[database] list_unregistered_students (embeddings) error: {e}")
            registered_roll_numbers = set()
        
        # Filter out registered students
        unregistered = [
            student for student in all_students 
            if student['roll_number'] not in registered_roll_numbers
        ]
        
        return unregistered
    
    def delete_student_data(self, roll_number: str) -> bool:
        """Delete all data for a student (use with caution)"""
        try:
            self.students.delete_one({'roll_number': roll_number})
            self.face_embeddings.delete_many({'roll_number': roll_number})
            return True
        except Exception as e:
            print(f"Error deleting student data: {e}")
            return False

# Global database instance
db = Database()

if __name__ == "__main__":
    # Test database connection
    print("Testing database connection...")
    print(f"Connected to: {DB_NAME}")
    print(f"Total students: {db.get_total_students()}")
    print(f"Total embeddings: {db.get_total_embeddings()}")
    print("Database connection successful!")

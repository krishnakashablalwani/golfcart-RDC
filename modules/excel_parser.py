"""
Excel Parser for Student Information
Reads student data from Excel file and imports into MongoDB
"""

import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import os

# Import from modules package
from .database import db

class StudentExcelParser:
    def __init__(self, excel_path: str = "data/Student information.xlsx"):
        """
        Initialize parser with Excel file path
        
        Args:
            excel_path: Path to the Excel file containing student information
        """
        self.excel_path = excel_path
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel file not found: {excel_path}")
        
        self.df = None
        self.load_excel()
    
    def load_excel(self):
        """Load Excel file into pandas DataFrame"""
        try:
            self.df = pd.read_excel(self.excel_path)
            # Ensure a 'registered' column exists; default to 'NO'
            if 'registered' not in [c.lower().strip() for c in self.df.columns]:
                # Keep original column names, but add a new one consistently named
                self.df['registered'] = 'NO'
            print(f"Successfully loaded Excel file with {len(self.df)} rows")
            print(f"Columns: {list(self.df.columns)}")
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            raise
    
    def parse_roll_number(self, roll_number: str) -> Dict[str, str]:
        """
        Parse roll number into components
        Format: 2451-25-733-075
        Returns: {year: '25', department_code: '733', class_roll: '075'}
        """
        parts = str(roll_number).strip().split('-')
        if len(parts) == 4:
            return {
                'year': parts[1],
                'department_code': parts[2],
                'class_roll': parts[3]
            }
        return {}
    
    def normalize_column_names(self):
        """Normalize column names to standard format"""
        # Common variations of column names
        column_mapping = {
            'roll number': 'roll_number',
            'rollnumber': 'roll_number',
            'roll_no': 'roll_number',
            'student name': 'name',
            'studentname': 'name',
            'full name': 'name',
            'student email': 'student_email',
            'email': 'student_email',
            'student phone': 'student_phone',
            'phone': 'student_phone',
            'mobile': 'student_phone',
            'parent phone': 'parent_phone',
            'father phone': 'father_phone',
            'mother phone': 'mother_phone',
            'parent email': 'parent_email',
            'parents email': 'parent_email',
            'department': 'department_name',
            'dept': 'department_name',
            'section': 'section',
            'year': 'academic_year',
            'academic year': 'academic_year',
            'registered': 'registered',
        }
        
        # Convert columns to lowercase and map
        self.df.columns = [col.lower().strip() for col in self.df.columns]
        self.df.rename(columns=column_mapping, inplace=True)
    
    def get_student_data(self, roll_number: str) -> Optional[Dict]:
        """
        Get student data by roll number from Excel
        
        Args:
            roll_number: Student roll number (e.g., 2451-25-733-075)
            
        Returns:
            Dictionary containing student information or None if not found
        """
        self.normalize_column_names()
        
        # Try to find student by roll number
        if 'roll_number' not in self.df.columns:
            print(f"Warning: 'roll_number' column not found. Available columns: {list(self.df.columns)}")
            return None
        
        # Search for roll number (handle different formats)
        mask = self.df['roll_number'].astype(str).str.strip() == str(roll_number).strip()
        student_rows = self.df[mask]
        
        if student_rows.empty:
            print(f"Student with roll number {roll_number} not found in Excel")
            return None
        
        # Get first matching row
        student = student_rows.iloc[0].to_dict()
        # Skip if already registered in Excel
        reg_val = str(student.get('registered', 'NO')).strip().upper()
        if reg_val == 'YES':
            return None
        
        # Parse roll number components
        roll_parts = self.parse_roll_number(roll_number)
        
        # Build student data dictionary
        student_data = {
            'roll_number': str(roll_number).strip(),
            'name': student.get('name', ''),
            'student_email': student.get('student_email', ''),
            'student_phone': student.get('student_phone', ''),
            'parent_phone': student.get('parent_phone', ''),
            'father_phone': student.get('father_phone', ''),
            'mother_phone': student.get('mother_phone', ''),
            'parent_email': student.get('parent_email', ''),
            'department_name': student.get('department_name', ''),
            'section': student.get('section', ''),
            'academic_year': student.get('academic_year', ''),
            **roll_parts  # Add parsed year, department_code, class_roll
        }
        
        # Remove None and NaN values
        student_data = {k: str(v).strip() if pd.notna(v) else '' 
                       for k, v in student_data.items()}
        
        return student_data

    def mark_registered_in_excel(self, roll_number: str) -> bool:
        """Mark a student as registered=YES in the Excel and save the file."""
        try:
            self.normalize_column_names()
            if 'roll_number' not in self.df.columns:
                return False
            mask = self.df['roll_number'].astype(str).str.strip() == str(roll_number).strip()
            if not mask.any():
                return False
            self.df.loc[mask, 'registered'] = 'YES'
            # Save back to the same file
            self.df.to_excel(self.excel_path, index=False)
            return True
        except Exception:
            return False
    
    def import_all_students(self) -> Dict[str, int]:
        """
        Import all students from Excel into MongoDB
        
        Returns:
            Dictionary with counts of successful and failed imports
        """
        self.normalize_column_names()
        
        if 'roll_number' not in self.df.columns:
            raise ValueError("Excel file must have a 'roll_number' column")
        
        stats = {
            'success': 0,
            'failed': 0,
            'skipped': 0
        }
        
        for idx, row in self.df.iterrows():
            roll_number = str(row.get('roll_number', '')).strip()
            
            if not roll_number or roll_number.lower() == 'nan':
                stats['skipped'] += 1
                continue
            
            # Respect Excel 'registered' flag
            reg_val = str(row.get('registered', 'NO')).strip().upper()
            if reg_val == 'YES':
                # Already registered, skip importing
                stats['skipped'] += 1
                continue

            # Check if student already exists
            if db.get_student(roll_number):
                print(f"Student {roll_number} already exists, skipping...")
                stats['skipped'] += 1
                continue
            
            student_data = self.get_student_data(roll_number)
            
            if student_data:
                if db.add_student(student_data):
                    print(f"Successfully imported: {roll_number} - {student_data.get('name', 'N/A')}")
                    stats['success'] += 1
                else:
                    print(f"Failed to import: {roll_number}")
                    stats['failed'] += 1
            else:
                stats['failed'] += 1
        
        return stats
    
    def get_all_roll_numbers(self) -> List[str]:
        """Get list of all roll numbers from Excel"""
        self.normalize_column_names()
        
        if 'roll_number' not in self.df.columns:
            return []
        
        # Only consider students present in Excel and not marked registered=YES
        rolls: List[str] = []
        for idx, row in self.df.iterrows():
            roll = row.get('roll_number')
            if pd.notna(roll) and str(roll).strip().lower() != 'nan':
                reg_val = str(row.get('registered', 'NO')).strip().upper()
                # Include even registered ones for lookup, but downstream can filter
                rolls.append(str(roll).strip())
        return rolls
    
    def search_student(self, query: str) -> List[Dict]:
        """
        Search for students by name or roll number
        
        Args:
            query: Search query (name or roll number)
            
        Returns:
            List of matching student records
        """
        self.normalize_column_names()
        query_lower = query.lower().strip()
        
        # Search in roll_number and name columns
        mask = (
            self.df.get('roll_number', pd.Series()).astype(str).str.lower().str.contains(query_lower, na=False) |
            self.df.get('name', pd.Series()).astype(str).str.lower().str.contains(query_lower, na=False)
        )
        
        results = []
        for idx, row in self.df[mask].iterrows():
            roll_number = str(row.get('roll_number', '')).strip()
            student_data = self.get_student_data(roll_number)
            if student_data:
                results.append(student_data)
        
        return results

def load_departments_from_file(filepath: str = "data/departments.txt"):
    """
    Load department mappings from a text file
    Format: CODE=Department Name
    Example: 733=Computer Science & Engineering
    """
    if not os.path.exists(filepath):
        print(f"Department file not found: {filepath}")
        return
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                code, name = line.split('=', 1)
                code = code.strip()
                name = name.strip()
                db.add_department(code, name)
                print(f"Added department: {code} = {name}")

if __name__ == "__main__":
    # Test the parser
    print("Testing Excel Parser...")
    
    try:
        parser = StudentExcelParser()
        
        # Show sample data
        print("\n--- Sample Roll Numbers ---")
        roll_numbers = parser.get_all_roll_numbers()[:5]
        for roll in roll_numbers:
            print(f"  {roll}")
        
        # Test parsing a specific student
        if roll_numbers:
            test_roll = roll_numbers[0]
            print(f"\n--- Testing Student Lookup: {test_roll} ---")
            student_data = parser.get_student_data(test_roll)
            if student_data:
                for key, value in student_data.items():
                    print(f"  {key}: {value}")
        
        # Ask if user wants to import all
        print(f"\n--- Found {len(roll_numbers)} students in Excel ---")
        response = input("Import all students to MongoDB? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            stats = parser.import_all_students()
            print(f"\n--- Import Complete ---")
            print(f"  Success: {stats['success']}")
            print(f"  Failed: {stats['failed']}")
            print(f"  Skipped: {stats['skipped']}")
    
    except Exception as e:
        print(f"Error: {e}")

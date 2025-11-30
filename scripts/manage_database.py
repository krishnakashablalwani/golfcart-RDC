"""
Utility Scripts for Managing 5K Students Database
Includes bulk import, export, and management tools
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.database import db
from modules.excel_parser import StudentExcelParser, load_departments_from_file
import pandas as pd
from datetime import datetime
import json

def import_all_students_from_excel():
    """Import all students from Excel file to MongoDB"""
    print("\n" + "="*60)
    print("BULK IMPORT STUDENTS FROM EXCEL")
    print("="*60 + "\n")
    
    try:
        parser = StudentExcelParser()
        
        # Show preview
        all_rolls = parser.get_all_roll_numbers()
        print(f"Found {len(all_rolls)} students in Excel file")
        
        if len(all_rolls) > 0:
            print("\nSample students:")
            for roll in all_rolls[:5]:
                student_data = parser.get_student_data(roll)
                if student_data:
                    print(f"  - {roll}: {student_data.get('name', 'N/A')}")
        
        # Confirm import
        response = input(f"\nImport all {len(all_rolls)} students to MongoDB? (yes/no): ")
        
        if response.lower() in ['yes', 'y']:
            print("\nImporting students...")
            stats = parser.import_all_students()
            
            print("\n" + "="*60)
            print("IMPORT COMPLETE")
            print("="*60)
            print(f"  Successfully imported: {stats['success']}")
            print(f"  Failed: {stats['failed']}")
            print(f"  Skipped (already exist): {stats['skipped']}")
            print(f"  Total in database: {db.get_total_students()}")
            print("="*60 + "\n")
        else:
            print("Import cancelled")
    
    except Exception as e:
        print(f"Error: {e}")

def import_departments():
    """Import department mappings from file"""
    print("\n" + "="*60)
    print("IMPORT DEPARTMENT MAPPINGS")
    print("="*60 + "\n")
    
    dept_file = input("Enter department file path (default: departments.txt): ").strip()
    if not dept_file:
        dept_file = "departments.txt"
    
    try:
        load_departments_from_file(dept_file)
        
        # Show loaded departments
        departments = db.get_all_departments()
        print(f"\nTotal departments in database: {len(departments)}")
        for dept in departments:
            print(f"  {dept['code']}: {dept['name']}")
    
    except Exception as e:
        print(f"Error: {e}")

def export_students_to_excel():
    """Export all students from MongoDB to Excel"""
    print("\n" + "="*60)
    print("EXPORT STUDENTS TO EXCEL")
    print("="*60 + "\n")
    
    try:
        # Get all students from database
        all_students = list(db.students.find())
        
        if not all_students:
            print("No students in database to export")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(all_students)
        
        # Remove MongoDB _id field
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"students_export_{timestamp}.xlsx"
        
        # Export to Excel
        df.to_excel(filename, index=False)
        
        print(f"Exported {len(all_students)} students to: {filename}")
    
    except Exception as e:
        print(f"Error: {e}")

def view_database_stats():
    """Display comprehensive database statistics"""
    print("\n" + "="*60)
    print("DATABASE STATISTICS")
    print("="*60 + "\n")
    
    # Overall stats
    total_students = db.get_total_students()
    total_embeddings = db.get_total_embeddings()
    total_detections = db.detections.count_documents({})
    
    print("Overall Statistics:")
    print(f"  Total Students: {total_students}")
    print(f"  Total Face Embeddings: {total_embeddings}")
    print(f"  Total Detections: {total_detections}")
    
    if total_students > 0:
        avg_samples = total_embeddings / total_students
        print(f"  Average Samples per Student: {avg_samples:.1f}")
    
    # Students by year
    print("\nStudents by Year:")
    years = db.students.distinct('year')
    for year in sorted(years):
        count = len(db.get_students_by_year(year))
        print(f"  Year 20{year}: {count} students")
    
    # Students by department
    print("\nStudents by Department:")
    dept_codes = db.students.distinct('department_code')
    for code in sorted(dept_codes):
        count = len(db.get_students_by_department(code))
        dept = db.get_department(code)
        dept_name = dept['name'] if dept else 'Unknown'
        print(f"  {code} ({dept_name}): {count} students")
    
    # Registration status
    print("\nRegistration Status:")
    students_with_embeddings = db.face_embeddings.distinct('roll_number')
    registered = len(students_with_embeddings)
    unregistered = total_students - registered
    print(f"  Registered (with face samples): {registered}")
    print(f"  Unregistered (no face samples): {unregistered}")
    
    # Recent detections
    print("\nRecent Detections (Last 24 hours):")
    today = datetime.now().date().isoformat()
    today_detections = db.get_detections(date=today, limit=10)
    
    if today_detections:
        print(f"  Total today: {len(today_detections)}")
        print("  Latest:")
        for detection in today_detections[:5]:
            student = db.get_student(detection['roll_number'])
            name = student['name'] if student else 'Unknown'
            timestamp = detection['timestamp'].strftime('%H:%M:%S')
            print(f"    {timestamp} - {name} ({detection['roll_number']}) - {detection['confidence']*100:.0f}%")
    else:
        print("  No detections today")
    
    print("\n" + "="*60 + "\n")

def search_student():
    """Search for a student by name or roll number"""
    print("\n" + "="*60)
    print("SEARCH STUDENT")
    print("="*60 + "\n")
    
    query = input("Enter name or roll number to search: ").strip()
    
    if not query:
        print("No search query entered")
        return
    
    # Try exact roll number first
    student = db.get_student(query)
    
    if student:
        display_student_details(student)
    else:
        # Search in Excel
        try:
            parser = StudentExcelParser()
            results = parser.search_student(query)
            
            if results:
                print(f"\nFound {len(results)} matching students:")
                for idx, student_data in enumerate(results, 1):
                    print(f"\n{idx}. {student_data['name']} - {student_data['roll_number']}")
                    
                    # Check if in database
                    db_student = db.get_student(student_data['roll_number'])
                    if db_student:
                        print("   Status: In database")
                        embeddings = db.get_student_embeddings(student_data['roll_number'])
                        print(f"   Face samples: {len(embeddings)}")
                    else:
                        print("   Status: Not in database")
            else:
                print("No matching students found")
        
        except Exception as e:
            print(f"Error searching: {e}")

def display_student_details(student):
    """Display detailed information about a student"""
    print("\n" + "-"*60)
    print("STUDENT DETAILS")
    print("-"*60)
    
    for key, value in student.items():
        if key not in ['_id', 'created_at', 'updated_at']:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Check face samples
    embeddings = db.get_student_embeddings(student['roll_number'])
    print(f"\n  Face Samples Registered: {len(embeddings)}")
    
    # Check detection history
    detections = db.get_detections(roll_number=student['roll_number'], limit=5)
    print(f"  Total Detections: {db.detections.count_documents({'roll_number': student['roll_number']})}")
    
    if detections:
        print("\n  Recent Detections:")
        for detection in detections[:5]:
            timestamp = detection['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            print(f"    - {timestamp}: {detection['confidence']*100:.0f}% confidence")
    
    print("-"*60 + "\n")

def list_unregistered_students():
    """List students who don't have face samples yet"""
    print("\n" + "="*60)
    print("UNREGISTERED STUDENTS (No Face Samples)")
    print("="*60 + "\n")
    
    # Get all students
    all_students = list(db.students.find())
    
    # Get students with embeddings
    students_with_embeddings = set(db.face_embeddings.distinct('roll_number'))
    
    # Find unregistered
    unregistered = []
    for student in all_students:
        if student['roll_number'] not in students_with_embeddings:
            unregistered.append(student)
    
    if unregistered:
        print(f"Found {len(unregistered)} unregistered students:\n")
        
        # Group by department
        by_dept = {}
        for student in unregistered:
            dept = student.get('department_code', 'Unknown')
            if dept not in by_dept:
                by_dept[dept] = []
            by_dept[dept].append(student)
        
        for dept_code in sorted(by_dept.keys()):
            dept = db.get_department(dept_code)
            dept_name = dept['name'] if dept else 'Unknown'
            print(f"\n{dept_code} - {dept_name}:")
            
            for student in by_dept[dept_code]:
                print(f"  - {student['roll_number']}: {student['name']}")
        
        # Option to export list
        print("\n" + "-"*60)
        response = input("\nExport list to text file? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            filename = f"unregistered_students_{datetime.now().strftime('%Y%m%d')}.txt"
            with open(filename, 'w') as f:
                f.write("Unregistered Students\n")
                f.write("="*60 + "\n\n")
                for student in unregistered:
                    f.write(f"{student['roll_number']}\t{student['name']}\t{student.get('department_name', 'N/A')}\n")
            print(f"Exported to: {filename}")
    else:
        print("All students are registered!")

def delete_student():
    """Delete a student and all associated data"""
    print("\n" + "="*60)
    print("DELETE STUDENT (CAUTION)")
    print("="*60 + "\n")
    
    roll_number = input("Enter student roll number to delete: ").strip()
    
    if not roll_number:
        print("No roll number entered")
        return
    
    # Check if student exists
    student = db.get_student(roll_number)
    
    if not student:
        print(f"Student {roll_number} not found in database")
        return
    
    # Show student details
    print(f"\nStudent: {student['name']}")
    print(f"Roll: {roll_number}")
    
    # Confirm deletion
    print("\nWARNING: This will delete:")
    print("  - Student record")
    print("  - All face samples and embeddings")
    print("  - Detection history will remain but be orphaned")
    
    confirm = input("\nType 'DELETE' to confirm: ").strip()
    
    if confirm == 'DELETE':
        if db.delete_student_data(roll_number):
            print(f"\nSuccessfully deleted student {roll_number}")
        else:
            print("\nError deleting student")
    else:
        print("\nDeletion cancelled")

def main_menu():
    """Main menu for database utilities"""
    while True:
        print("\n" + "="*60)
        print("DATABASE MANAGEMENT UTILITIES")
        print("="*60)
        print("\n1. Import all students from Excel")
        print("2. Import department mappings")
        print("3. Export students to Excel")
        print("4. View database statistics")
        print("5. Search student")
        print("6. List unregistered students")
        print("7. Delete student")
        print("8. Exit")
        
        choice = input("\nEnter choice (1-8): ").strip()
        
        if choice == '1':
            import_all_students_from_excel()
        elif choice == '2':
            import_departments()
        elif choice == '3':
            export_students_to_excel()
        elif choice == '4':
            view_database_stats()
        elif choice == '5':
            search_student()
        elif choice == '6':
            list_unregistered_students()
        elif choice == '7':
            delete_student()
        elif choice == '8':
            print("\nExiting...")
            break
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("GOLF CART FACE RECOGNITION - DATABASE UTILITIES")
    print("Student Management System for 5K Students")
    print("="*60)
    
    main_menu()

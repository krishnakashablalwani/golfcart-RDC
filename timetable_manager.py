import json
import datetime
import os
import pandas as pd

TIMETABLE_FILE = 'timetable.json'
STUDENTS_FILE = 'students.xlsx'

def load_timetable():
    if not os.path.exists(TIMETABLE_FILE):
        return {}
    with open(TIMETABLE_FILE, 'r') as f:
        return json.load(f)

def get_student_class(roll_number):
    """
    Looks up the student's class/section from students.xlsx.
    Assumes there is a 'class' or 'section' column.
    If not found, defaults to 'CSE-A' for demo purposes.
    """
    if not os.path.exists(STUDENTS_FILE):
        return "CSE-A"
    
    try:
        df = pd.read_excel(STUDENTS_FILE)
        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Find roll number column
        roll_col = next((c for c in df.columns if 'roll' in c), None)
        class_col = next((c for c in df.columns if 'class' in c or 'section' in c), None)
        
        if roll_col and class_col:
            student = df[df[roll_col].astype(str) == str(roll_number)]
            if not student.empty:
                return student.iloc[0][class_col]
    except Exception as e:
        print(f"Error reading student class: {e}")
    
    return "CSE-A" # Default for demo

def get_parent_email(roll_number):
    """
    Looks up parent email from students.xlsx
    """
    if not os.path.exists(STUDENTS_FILE):
        return None
        
    try:
        df = pd.read_excel(STUDENTS_FILE)
        df.columns = [c.strip().lower() for c in df.columns]
        
        roll_col = next((c for c in df.columns if 'roll' in c), None)
        email_col = next((c for c in df.columns if 'parent' in c and 'email' in c), None)
        
        if roll_col and email_col:
            student = df[df[roll_col].astype(str) == str(roll_number)]
            if not student.empty:
                return student.iloc[0][email_col]
    except Exception:
        pass
    return None

def check_class_status(roll_number):
    """
    Checks if the student should be in class right now.
    Returns: (bool, str) -> (is_busy, subject_name)
    """
    student_class = get_student_class(roll_number)
    timetable = load_timetable()
    
    if student_class not in timetable:
        return False, None
        
    now = datetime.datetime.now()
    day_name = now.strftime("%A") # e.g., "Monday"
    current_time = now.strftime("%H:%M")
    
    todays_schedule = timetable[student_class].get(day_name, [])
    
    for slot in todays_schedule:
        if slot['start'] <= current_time <= slot['end']:
            return True, slot['subject']
            
    return False, None

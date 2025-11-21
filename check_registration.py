import pandas as pd
from pymongo import MongoClient
import os

# MongoDB settings
MONGODB_URI = "mongodb://localhost:27017/"
DB_NAME = "face_recognition_db"
COLLECTION_NAME = "registered_faces_ultra"  # uses 'roll_number' field when registering faces


def get_database():
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    return collection


ROLL_COLUMN_CANDIDATES = [
    "roll_number",
    "roll number",
    "roll no",
    "roll",
    "rollno",
    "roll_no",
]
NAME_COLUMN_CANDIDATES = ["name", "student name", "full name"]


def standardize_columns(df):
    """Return tuple (roll_col, name_col) after finding acceptable column names.
    Performs case-insensitive, whitespace-stripped matching.
    """
    normalized = {c: c for c in df.columns}
    # Build lookup map from simplified key -> original column
    simplemap = {}
    for c in df.columns:
        key = c.strip().lower().replace("_", " ")
        simplemap[key] = c

    def find(col_candidates):
        for cand in col_candidates:
            cand_key = cand.strip().lower().replace("_", " ")
            if cand_key in simplemap:
                return simplemap[cand_key]
        return None

    roll_col = find(ROLL_COLUMN_CANDIDATES)
    name_col = find(NAME_COLUMN_CANDIDATES)
    return roll_col, name_col


def check_registration():
    excel_file = "students.xlsx"
    if not os.path.exists(excel_file):
        print(f"File not found: {excel_file}")
        return

    try:
        df = pd.read_excel(excel_file)
    except Exception as e:
        print(f"Unable to read Excel file: {e}")
        return

    print(f"Loaded {len(df)} records from Excel.")

    roll_col, name_col = standardize_columns(df)
    if roll_col is None or name_col is None:
        print("Could not find required columns. Need a roll number column (e.g. 'Roll no') and 'Name'.")
        print(f"Columns found: {list(df.columns)}")
        return

    # Normalize roll numbers to string to match DB usage
    df[roll_col] = df[roll_col].astype(str).str.strip()
    df[name_col] = df[name_col].astype(str).str.strip()

    collection = get_database()
    registered_docs = collection.find({}, {"roll_number": 1, "_id": 0})
    registered_rolls = {str(doc.get("roll_number")).strip() for doc in registered_docs}
    print(f"Found {len(registered_rolls)} registered faces in database.")

    # Determine unregistered
    unregistered_mask = ~df[roll_col].isin(registered_rolls)
    unregistered = df[unregistered_mask].copy()

    total = len(df)
    missing = len(unregistered)
    percent = (total - missing) / total * 100 if total else 0
    print(f"\nProgress: {total - missing}/{total} registered ({percent:.1f}%). Missing: {missing}")

    if missing == 0:
        print("All students are registered! ✅")
        return

    # Show first 10 unregistered
    print(f"\nUnregistered students (showing up to 10 of {missing}):")
    for _, row in unregistered.head(10).iterrows():
        print(f"  Roll {row[roll_col]} - {row[name_col]}")

    # Next to register (smallest lexicographic roll)
    try:
        # Attempt numeric sort if possible
        numeric_rolls = pd.to_numeric(unregistered[roll_col], errors="coerce")
        if numeric_rolls.notna().any():
            next_index = numeric_rolls.idxmin()
        else:
            next_index = unregistered[roll_col].idxmin()
        next_roll = unregistered.loc[next_index, roll_col]
        next_name = unregistered.loc[next_index, name_col]
    except Exception:
        # Fallback to first row
        next_roll = unregistered.iloc[0][roll_col]
        next_name = unregistered.iloc[0][name_col]

    print(f"\nNext to register: Roll {next_roll} - {next_name}")

    # Optional export of full missing list
    export_name = "unregistered_faces.csv"
    try:
        unregistered[[roll_col, name_col]].to_csv(export_name, index=False)
        print(f"Full missing list exported to {export_name}")
    except Exception as e:
        print(f"Could not export CSV: {e}")

    # Group summary by department/year if those columns exist
    extra_cols = {c.lower(): c for c in df.columns}
    dept_col = extra_cols.get("department")
    year_col = extra_cols.get("year")
    sem_col = extra_cols.get("sem")
    if dept_col and dept_col in unregistered.columns:
        print("\nMissing by Department (top 10):")
        print(unregistered.groupby(dept_col)[roll_col].count().sort_values(ascending=False).head(10))
    if year_col and year_col in unregistered.columns:
        print("\nMissing by Year:")
        print(unregistered.groupby(year_col)[roll_col].count())
    if sem_col and sem_col in unregistered.columns:
        print("\nMissing by Sem:")
        print(unregistered.groupby(sem_col)[roll_col].count())


if __name__ == "__main__":
    check_registration()
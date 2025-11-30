"""
Clear Face Recognition Database
Removes all face embeddings, sample images, detection logs, and cache files
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import shutil
from modules.database import db

def clear_database():
    """Clear all face recognition data"""
    
    print("\n" + "="*60)
    print("CLEAR FACE RECOGNITION DATABASE")
    print("="*60)
    print("\nThis will delete:")
    print("  - All face embeddings from MongoDB")
    print("  - All sample images (Samples/ folder)")
    print("  - All detection logs (Detections/ folder)")
    print("  - Embedding cache file")
    print("\n⚠️  WARNING: This action cannot be undone!")
    print("="*60)
    
    # Confirm
    confirm = input("\nType 'YES' to confirm deletion: ").strip()
    
    if confirm != 'YES':
        print("❌ Cancelled. No data was deleted.")
        return
    
    print("\nClearing database...")
    
    # Delete face embeddings from MongoDB
    result_embeddings = db.face_embeddings.delete_many({})
    print(f"✅ Deleted {result_embeddings.deleted_count} face embeddings from MongoDB")
    
    # Delete Samples folder
    samples_dir = Path('Samples')
    if samples_dir.exists():
        shutil.rmtree(samples_dir)
        print("✅ Deleted Samples/ folder")
    else:
        print("ℹ️  Samples/ folder not found")
    
    # Delete Detections folder
    detections_dir = Path('Detections')
    if detections_dir.exists():
        shutil.rmtree(detections_dir)
        print("✅ Deleted Detections/ folder")
    else:
        print("ℹ️  Detections/ folder not found")
    
    # Delete cache file
    cache_file = Path('face_embeddings_cache.pkl')
    if cache_file.exists():
        cache_file.unlink()
        print("✅ Deleted embedding cache file")
    else:
        print("ℹ️  Cache file not found")
    
    # Optional: Clear detection logs (uncomment if needed)
    # result_detections = db.detections.delete_many({})
    # print(f"✅ Deleted {result_detections.deleted_count} detection logs")
    
    print("\n" + "="*60)
    print("✅ DATABASE CLEARED SUCCESSFULLY!")
    print("="*60)
    
    # Show remaining students
    unregistered = db.list_unregistered_students()
    print(f"\n📋 {len(unregistered)} students ready to register")
    print("\nRun 'python register_face_deepface.py' to start registration")
    print("="*60 + "\n")

if __name__ == "__main__":
    clear_database()

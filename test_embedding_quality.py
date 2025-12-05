""".venv/Scripts/python.exe test_camera.py  
Test embedding consistency and quality
"""
import numpy as np
import os.path as osp
from pathlib import Path
import cv2
import mediapipe as mp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load InsightFace
try:
    from insightface.model_zoo.arcface_onnx import ArcFaceONNX
    
    model_path = osp.join(osp.expanduser("~"), ".insightface", "models", "buffalo_sc", "w600k_mbf.onnx")
    arcface_model = ArcFaceONNX(model_file=model_path)
    arcface_model.prepare(ctx_id=-1)  # CPU only for testing
    print(f"✓ ArcFace loaded")
except Exception as e:
    logger.error(f"Failed to load InsightFace: {e}")
    exit(1)

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Load stored embedding
from modules.database import db

def test_embedding_quality():
    """Test consistency of stored embedding"""
    try:
        # Get Krishna's data
        doc = db.face_embeddings.find_one({"roll_number": "2451-25-733-075"})
        if not doc:
            print("❌ No embedding found for Krishna")
            return
        
        master_emb = np.array(doc['embedding'], dtype=np.float32)
        print(f"Master embedding loaded - norm: {np.linalg.norm(master_emb):.4f}, shape: {master_emb.shape}")
        
        # Load sample images
        sample_dir = Path("Samples") / "2025" / "733" / "2451-25-733-075"
        if not sample_dir.exists():
            print(f"❌ Sample directory not found: {sample_dir}")
            return
        
        images = list(sample_dir.glob("*.jpg"))
        if not images:
            print(f"❌ No sample images found")
            return
        
        print(f"\n📷 Testing {len(images)} sample images")
        print("="*70)
        
        embeddings = []
        distances_to_master = []
        
        for i, img_path in enumerate(sorted(images)[:10]):  # Test first 10
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            
            # Get embedding (same as registration code)
            face_112 = cv2.resize(frame, (112, 112))
            emb = arcface_model.get_feat(face_112)
            emb = np.array(emb, dtype=np.float32)
            
            # L2 normalize to unit norm (CRITICAL - must match registration)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            
            embeddings.append(emb)
            
            # Distance to master
            dist_to_master = np.linalg.norm(emb - master_emb)
            distances_to_master.append(dist_to_master)
            
            print(f"Sample {i+1:2d}: Dist to master = {dist_to_master:.4f}, Norm = {np.linalg.norm(emb):.4f}")
        
        print("="*70)
        print(f"Mean distance to master: {np.mean(distances_to_master):.4f}")
        print(f"Max distance to master:  {np.max(distances_to_master):.4f}")
        print(f"Min distance to master:  {np.min(distances_to_master):.4f}")
        
        # Test consistency between samples
        print("\n📊 Sample-to-sample distances (first 5):")
        print("="*70)
        for i in range(min(5, len(embeddings))):
            for j in range(i+1, min(5, len(embeddings))):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                print(f"Sample {i+1} <-> Sample {j+1}: {dist:.4f}")
        
        # Expected confidence
        print("\n📈 Expected confidence at different distances:")
        print("="*70)
        test_distances = [0.2, 0.3, 0.4, 0.5, 0.6]
        for d in test_distances:
            if d < 0.25:
                conf = 0.99
            elif d < 0.35:
                conf = 0.95
            elif d < 0.45:
                conf = 0.88
            elif d < 0.55:
                conf = 0.78
            else:
                conf = 0.65
            print(f"Distance {d:.2f} → Confidence {conf:.1%}")
        
        print("\n⚠️  If your actual distances are > 0.5, the embeddings are inconsistent!")
        print("    This means the camera quality or lighting is varying too much during registration.")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_embedding_quality()
